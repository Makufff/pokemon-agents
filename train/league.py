# train/league.py
import copy
import math
import random

import torch
import torch.nn.functional as F

from core.env import PTCGEnv
from model.net import PolicyValueNet
from train.buffer import LearnSample, RingBuffer
from train.dmc import (
    mcts_step, apply_td_lambda, _pad,
    MAX_PAD_HAND, MAX_PAD_DISCARD, MAX_PAD_DECK,
    _eval_obs,
)
from train.ppo import compute_ppo_loss, compute_upgo_returns, compute_kl_loss

MAIN_VS_MAIN_PROB = 0.70
MAIN_VS_EXPLOITER_PROB = 0.20
KL_WEIGHT_INIT = 0.01
EXPLOITER_DOMINANCE_THRESHOLD = 0.60


class League:
    def __init__(self, main_agent, exploiter_agent, teacher, deck, device, kl_weight=KL_WEIGHT_INIT):
        self.main = main_agent
        self.exploiter = exploiter_agent
        self.teacher = teacher
        self.deck = deck
        self.device = device
        self.kl_weight = kl_weight
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def sample_matchup(self):
        r = random.random()
        if r < MAIN_VS_MAIN_PROB:
            return 'main_vs_main'
        elif r < MAIN_VS_MAIN_PROB + MAIN_VS_EXPLOITER_PROB:
            return 'main_vs_exploiter'
        return 'main_vs_teacher'

    def play_game(self, matchup='main_vs_main', search_count=5, epsilon=0.1):
        """Play one game. Returns LearnSamples for main agent turns only."""
        model_map = {
            'main_vs_main': (self.main, self.main),
            'main_vs_exploiter': (self.main, self.exploiter),
            'main_vs_teacher': (self.main, self.teacher),
        }
        if matchup not in model_map:
            raise ValueError(f"Unknown matchup: {matchup!r}")
        p0_model, p1_model = model_map[matchup]
        models = [p0_model, p1_model]

        env = PTCGEnv()
        obs = env.reset(self.deck, self.deck, your_index=0)
        samples_by_player = [[], []]
        hand_cache = {0: [], 1: []}
        done = False

        while not done:
            sel = obs.get('select')
            if sel is None:
                obs, done, _ = env.step([])
                continue

            acting = obs['current']['yourIndex']
            acting_hand = obs['current']['players'][acting].get('hand') or []
            hand_cache[acting] = [c['id'] for c in acting_hand if c]
            model_for_acting = models[acting]

            if random.random() < epsilon:
                n_opts = len(sel['option'])
                mc = sel['maxCount']
                action = random.sample(range(n_opts), min(mc, n_opts))
                _, _, probs, sample = _eval_obs(obs, acting, self.deck, model_for_acting, self.device)
                if sample is not None:
                    p = max(probs[0] if probs else 1e-8, 1e-8)
                    sample.log_prob_old = math.log(p)
                    sample.opp_hand_ids = hand_cache[1 - acting]
                    samples_by_player[acting].append(sample)
            else:
                action, sample = mcts_step(obs, self.deck, model_for_acting, self.device, search_count)
                if sample is not None:
                    sample.opp_hand_ids = hand_cache[1 - acting]
                    samples_by_player[acting].append(sample)

            obs, done, _ = env.step(action)

        env.close()
        result = obs['current']['result']

        all_main_samples = []
        for pi in range(2):
            if models[pi] is self.main:
                terminal = 1.0 if result == pi else (-1.0 if result != 2 else 0.0)
                apply_td_lambda(samples_by_player[pi], result, your_index=pi)
                upgo_rets = compute_upgo_returns(samples_by_player[pi], terminal)
                for s, ur in zip(samples_by_player[pi], upgo_rets):
                    s.td_value = ur
                all_main_samples.extend(samples_by_player[pi])

        return all_main_samples

    def update_exploiter(self):
        self.exploiter.load_state_dict(copy.deepcopy(self.main.state_dict()))

    def eval_exploiter_vs_main(self, n_games=20):
        wins = 0
        for i in range(n_games):
            your_index = i % 2
            env = PTCGEnv()
            obs = env.reset(self.deck, self.deck, your_index=your_index)
            done = False
            while not done:
                sel = obs.get('select')
                if sel is None:
                    obs, done, _ = env.step([])
                    continue
                acting = obs['current']['yourIndex']
                model_for_acting = self.exploiter if acting == your_index else self.main
                with torch.no_grad():
                    action, _ = mcts_step(obs, self.deck, model_for_acting, self.device, search_count=3)
                obs, done, _ = env.step(action)
            env.close()
            if obs['current']['result'] == your_index:
                wins += 1
        return wins / n_games

    def ppo_train_step(self, batch, optimizer, upgo_returns, clip_eps=0.2):
        """One PPO+UPGO gradient step. Returns total loss as float."""
        self.main.train()
        losses = []

        for sample, upgo_ret in zip(batch, upgo_returns):
            board = torch.tensor(sample.board, device=self.device).unsqueeze(0)
            hand_t = _pad(sample.hand_ids, MAX_PAD_HAND).unsqueeze(0).to(self.device)
            discard_t = _pad(sample.discard_ids, MAX_PAD_DISCARD).unsqueeze(0).to(self.device)
            deck_t = _pad(sample.deck_ids, MAX_PAD_DECK).unsqueeze(0).to(self.device)
            scalars = torch.tensor(sample.scalars, device=self.device).unsqueeze(0)
            opt_types = torch.tensor(sample.opt_types, dtype=torch.long, device=self.device)
            opt_cards = torch.tensor(sample.opt_cards, dtype=torch.long, device=self.device)

            opp_hand_t = None
            if sample.opp_hand_ids:
                opp_hand_t = _pad(sample.opp_hand_ids, MAX_PAD_HAND).unsqueeze(0).to(self.device)

            value, scores = self.main(
                board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards, opp_hand_t
            )

            v_target = torch.tensor([[upgo_ret]], dtype=torch.float32, device=self.device)
            loss_v = F.huber_loss(value, v_target, delta=0.2)

            log_probs = F.log_softmax(scores, dim=0)
            new_log_prob = log_probs[sample.action_idx]
            advantage = upgo_ret - value.detach().item()
            loss_p = compute_ppo_loss(new_log_prob, sample.log_prob_old, advantage, clip_eps)

            if self.kl_weight > 1e-6:
                kl = compute_kl_loss(self.main, self.teacher, sample, self.device)
                total = loss_v + loss_p + self.kl_weight * kl
            else:
                total = loss_v + loss_p

            losses.append(total)

        combined = torch.stack(losses).mean()
        optimizer.zero_grad()
        combined.backward()
        torch.nn.utils.clip_grad_norm_(self.main.parameters(), 1.0)
        optimizer.step()
        return abs(combined.item())


def run_league(
    deck, main_agent, exploiter_agent, teacher, optimizer, device,
    save_path='model_phase2.pt',
    n_iterations=1000, games_per_iter=5, batch_size=32, buffer_capacity=50_000,
    epsilon_start=0.3, epsilon_end=0.05, search_count=5,
    eval_every=50, eval_games=30,
    exploiter_update_every=200,
    kl_weight_init=KL_WEIGHT_INIT, kl_anneal_iters=750,
):
    """Phase 2 outer training loop."""
    from train.dmc import eval_vs_random

    league = League(main_agent, exploiter_agent, teacher, deck, device, kl_weight=kl_weight_init)
    buf = RingBuffer(capacity=buffer_capacity)

    for iteration in range(n_iterations):
        eps = epsilon_start + (epsilon_end - epsilon_start) * iteration / n_iterations
        league.kl_weight = kl_weight_init * max(0.0, 1.0 - iteration / kl_anneal_iters)

        main_agent.eval()
        for _ in range(games_per_iter):
            matchup = league.sample_matchup()
            samples = league.play_game(matchup=matchup, search_count=search_count, epsilon=eps)
            for s in samples:
                buf.push(s)

        if len(buf) >= batch_size:
            batch = buf.sample(batch_size)
            upgo_returns = [s.td_value for s in batch]
            loss = league.ppo_train_step(batch, optimizer, upgo_returns)
            print(f"iter {iteration:4d} | buf {len(buf):6d} | loss {loss:.4f} | kl_w {league.kl_weight:.5f} | eps {eps:.3f}")

        if (iteration + 1) % exploiter_update_every == 0:
            league.update_exploiter()
            exp_wr = league.eval_exploiter_vs_main(n_games=10)
            print(f"  exploiter vs main: {exp_wr:.1%}")
            if exp_wr > EXPLOITER_DOMINANCE_THRESHOLD:
                league.kl_weight = min(league.kl_weight * 2, 0.10)
                print(f"  KL weight bumped to {league.kl_weight:.5f}")

        if (iteration + 1) % eval_every == 0:
            wr = eval_vs_random(deck, main_agent, device, n_games=eval_games)
            print(f"  >> main win rate vs random: {wr:.1%}")
            torch.save({'model': main_agent.state_dict(), 'iteration': iteration}, save_path)
