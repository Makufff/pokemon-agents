# train_phase2.py
"""Phase 2 PPO+UPGO League training with multi-GPU async self-play.

GPU 0: main training loop + self-play
GPU N: async self-play workers (one per extra GPU)

Run: python -u train_phase2.py > logs/train_phase2.log 2>&1
"""
import copy
import os
import queue

import torch
import torch.multiprocessing as mp

from model.net import PolicyValueNet
from train.league import League, EXPLOITER_DOMINANCE_THRESHOLD
from train.buffer import RingBuffer
from train.dmc import eval_vs_random

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DMC_CKPT = 'model.pt'
SAVE_PATH = 'model_phase2.pt'

N_ITERATIONS = 2000
GAMES_PER_ITER = 5
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
EPSILON_START = 0.3
EPSILON_END = 0.05
SEARCH_COUNT = 5
EVAL_EVERY = 50
EVAL_GAMES = 30
EXPLOITER_UPDATE_EVERY = 200
KL_WEIGHT_INIT = 0.01
KL_ANNEAL_ITERS = 1500
WEIGHT_SYNC_EVERY = 5  # trainer iters between weight broadcasts to workers


def _worker_fn(gpu_id: int, deck: list, sample_q: mp.Queue,
               weight_q: mp.Queue, stop_ev: mp.Event) -> None:
    """Async self-play worker. Runs on gpu_id, emits LearnSamples into sample_q."""
    device = torch.device(f'cuda:{gpu_id}')

    model = PolicyValueNet().to(device)
    model.eval()
    exploiter = copy.deepcopy(model)
    teacher = copy.deepcopy(model)

    league = League(model, exploiter, teacher, deck, device, kl_weight=0.0)

    iteration = 0
    while not stop_ev.is_set():
        # Pull latest weights (drain queue, keep only newest)
        latest_sd = None
        while True:
            try:
                latest_sd = weight_q.get_nowait()
            except Exception:
                break
        if latest_sd is not None:
            model.load_state_dict({k: v.to(device) for k, v in latest_sd.items()})
            model.eval()

        eps = max(EPSILON_END,
                  EPSILON_START + (EPSILON_END - EPSILON_START) * iteration / N_ITERATIONS)

        for _ in range(GAMES_PER_ITER):
            matchup = league.sample_matchup()
            try:
                samples = league.play_game(matchup=matchup,
                                           search_count=SEARCH_COUNT,
                                           epsilon=eps)
                for s in samples:
                    sample_q.put(s)
            except Exception:
                pass

        iteration += 1


def _drain(q: mp.Queue, buf: RingBuffer, limit: int = 2000) -> int:
    """Move up to limit items from q into buf. Returns count moved."""
    count = 0
    while count < limit:
        try:
            s = q.get_nowait()
            buf.push(s)
            count += 1
        except Exception:
            break
    return count


def _load_dmc_checkpoint(model: PolicyValueNet, device: torch.device) -> None:
    if not os.path.exists(DMC_CKPT):
        return
    try:
        ckpt = torch.load(DMC_CKPT, map_location=device, weights_only=False)
        current_shapes = {k: v.shape for k, v in model.state_dict().items()}
        filtered = {k: v for k, v in ckpt['model'].items()
                    if k in current_shapes and v.shape == current_shapes[k]}
        skipped = len(ckpt['model']) - len(filtered)
        result = model.load_state_dict(filtered, strict=False)
        print(f'Loaded DMC checkpoint '
              f'(missing={len(result.missing_keys)}, skipped_shape={skipped})')
    except Exception as e:
        print(f'Could not load DMC checkpoint ({e}), starting from scratch')


def main() -> None:
    mp.set_start_method('spawn', force=True)

    n_gpus = torch.cuda.device_count()
    print(f'Training on {n_gpus} GPU(s)')

    device = torch.device('cuda:0')

    main_agent = PolicyValueNet().to(device)
    _load_dmc_checkpoint(main_agent, device)

    teacher = copy.deepcopy(main_agent)
    exploiter = copy.deepcopy(main_agent)
    optimizer = torch.optim.AdamW(main_agent.parameters(), lr=1e-4, weight_decay=1e-4)

    # Launch async workers on GPU 1..N-1
    sample_q: mp.Queue = mp.Queue(maxsize=10_000)
    weight_qs: list[mp.Queue] = []
    stop_ev = mp.Event()
    workers: list[mp.Process] = []

    for gpu_id in range(1, n_gpus):
        wq: mp.Queue = mp.Queue(maxsize=10)
        weight_qs.append(wq)
        p = mp.Process(
            target=_worker_fn,
            args=(gpu_id, DECK, sample_q, wq, stop_ev),
            daemon=True,
        )
        p.start()
        workers.append(p)
        print(f'Started worker on GPU {gpu_id} (pid={p.pid})')

    buf = RingBuffer(capacity=BUFFER_CAPACITY)
    league = League(main_agent, exploiter, teacher, DECK, device, kl_weight=KL_WEIGHT_INIT)

    try:
        for iteration in range(N_ITERATIONS):
            eps = EPSILON_START + (EPSILON_END - EPSILON_START) * iteration / N_ITERATIONS
            league.kl_weight = KL_WEIGHT_INIT * max(0.0, 1.0 - iteration / KL_ANNEAL_ITERS)

            # Drain worker samples accumulated so far
            _drain(sample_q, buf)

            # Local self-play on GPU 0
            main_agent.eval()
            for _ in range(GAMES_PER_ITER):
                matchup = league.sample_matchup()
                try:
                    samples = league.play_game(matchup=matchup,
                                               search_count=SEARCH_COUNT,
                                               epsilon=eps)
                    for s in samples:
                        buf.push(s)
                except Exception:
                    pass

            # Drain worker samples that arrived during local play
            _drain(sample_q, buf)

            # Training step
            if len(buf) >= BATCH_SIZE:
                batch = buf.sample(BATCH_SIZE)
                upgo_returns = [s.td_value for s in batch]
                loss = league.ppo_train_step(batch, optimizer, upgo_returns)
                print(f'iter {iteration:4d} | buf {len(buf):6d} | loss {loss:.4f} '
                      f'| kl_w {league.kl_weight:.5f} | eps {eps:.3f}')

            # Broadcast updated weights to workers
            if weight_qs and iteration % WEIGHT_SYNC_EVERY == 0:
                state_cpu = {k: v.cpu() for k, v in main_agent.state_dict().items()}
                for wq in weight_qs:
                    # Drain stale weight to avoid backpressure
                    while not wq.empty():
                        try:
                            wq.get_nowait()
                        except Exception:
                            break
                    try:
                        wq.put_nowait(state_cpu)
                    except Exception:
                        pass

            # Exploiter update
            if (iteration + 1) % EXPLOITER_UPDATE_EVERY == 0:
                league.update_exploiter()
                exp_wr = league.eval_exploiter_vs_main(n_games=10)
                print(f'  exploiter vs main: {exp_wr:.1%}')
                if exp_wr > EXPLOITER_DOMINANCE_THRESHOLD:
                    league.kl_weight = min(league.kl_weight * 2, 0.10)
                    print(f'  KL weight bumped to {league.kl_weight:.5f}')

            # Eval + checkpoint
            if (iteration + 1) % EVAL_EVERY == 0:
                wr = eval_vs_random(DECK, main_agent, device, n_games=EVAL_GAMES)
                print(f'  >> main win rate vs random: {wr:.1%}')
                torch.save({'model': main_agent.state_dict(), 'iteration': iteration},
                           SAVE_PATH)

    finally:
        stop_ev.set()
        for p in workers:
            p.join(timeout=15)
            if p.is_alive():
                p.kill()
        print('All workers stopped.')


if __name__ == '__main__':
    main()
