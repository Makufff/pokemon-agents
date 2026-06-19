# train/dmc.py
import dataclasses
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from cg.api import search_begin, search_step, search_end, to_observation_class
from core.env import PTCGEnv
from core.features import (
    encode_board, encode_sets, encode_scalars, enumerate_actions, encode_option,
    CARD_COUNT,
)
from model.net import PolicyValueNet
from train.buffer import LearnSample, RingBuffer


MAX_PAD_HAND = 20
MAX_PAD_DISCARD = 60
MAX_PAD_DECK = 60
PUCT_C = 0.4


def _pad(ids: list[int], max_len: int) -> torch.Tensor:
    arr = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return torch.tensor(arr, dtype=torch.long)


def obs_to_tensors(obs: dict, your_deck: list[int], device: torch.device):
    """Convert obs dict to model input tensors (batch size 1)."""
    your_index = obs['current']['yourIndex']
    board = torch.tensor(encode_board(obs, your_index), dtype=torch.float32, device=device).unsqueeze(0)
    hand_ids, discard_ids, deck_ids = encode_sets(obs, your_index, your_deck)
    hand_t = _pad(hand_ids, MAX_PAD_HAND).unsqueeze(0).to(device)
    discard_t = _pad(discard_ids, MAX_PAD_DISCARD).unsqueeze(0).to(device)
    deck_t = _pad(deck_ids, MAX_PAD_DECK).unsqueeze(0).to(device)
    scalars = torch.tensor(encode_scalars(obs, your_index), dtype=torch.float32, device=device).unsqueeze(0)
    return board, hand_t, discard_t, deck_t, scalars, hand_ids, discard_ids, deck_ids


def _encode_actions(obs: dict, actions: list[list[int]], your_index: int, device: torch.device):
    """Return (opt_types [N], opt_cards [N]) tensors for candidate actions."""
    sel = obs.get('select') or {}
    options = sel.get('option', [])
    type_ids, card_ids = [], []
    for action in actions:
        if not action:
            type_ids.append(0); card_ids.append(0)
            continue
        idx = action[0]
        if idx < len(options):
            t, c = encode_option(options[idx], obs, your_index)
        else:
            t, c = 0, 0
        type_ids.append(t); card_ids.append(c)
    return (
        torch.tensor(type_ids, dtype=torch.long, device=device),
        torch.tensor(card_ids, dtype=torch.long, device=device),
    )


def _obs_class_to_dict(obs_class) -> dict:
    """Convert an Observation dataclass to a plain dict (for feature encoding)."""
    return json.loads(json.dumps(
        dataclasses.asdict(obs_class),
        default=lambda o: o.value if hasattr(o, 'value') else str(o),
    ))


@dataclass
class _Node:
    value: float
    total: float
    visit: int
    children: list


@dataclass
class _Child:
    action: list[int]
    prob: float
    node: object  # _Node | None
    search_id: int


def _eval_obs(obs_raw: dict, your_index: int, your_deck: list[int],
              model: PolicyValueNet, device: torch.device):
    """Run model on obs_raw, return (value_float, actions, probs, sample)."""
    state = obs_raw.get('current') or {}
    result = state.get('result', -1)

    if result >= 0:
        v = 1.0 if result == your_index else (-1.0 if result != 2 else 0.0)
        return v, [], [], None

    board, hand_t, discard_t, deck_t, scalars, hand_ids, discard_ids, deck_ids = \
        obs_to_tensors(obs_raw, your_deck, device)
    actions = enumerate_actions(obs_raw)
    if not actions:
        return 0.0, [], [], None

    opt_types, opt_cards = _encode_actions(obs_raw, actions, your_index, device)

    with torch.no_grad():
        value, scores = model(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)

    v = value.item()
    # Flip sign if it's the opponent's turn
    acting = state.get('yourIndex', your_index)
    if acting != your_index:
        v = -v

    probs = torch.softmax(scores, dim=0).cpu().tolist()

    sample = LearnSample(
        board=board.squeeze(0).cpu().numpy(),
        hand_ids=hand_ids,
        discard_ids=discard_ids,
        deck_ids=deck_ids,
        scalars=scalars.squeeze(0).cpu().numpy(),
        opt_types=[t.item() for t in opt_types],
        opt_cards=[c.item() for c in opt_cards],
        action_idx=0,
        td_value=v,
        mcts_policy=probs,
    )
    return v, actions, probs, sample


def mcts_step(obs_dict: dict, your_deck: list[int], model: PolicyValueNet,
              device: torch.device, search_count: int = 10) -> tuple[list[int], 'LearnSample | None']:
    """Run MCTS from obs_dict, return (selected_action, LearnSample)."""
    obs_class = to_observation_class(obs_dict)
    state = obs_class.current
    your_index = state.yourIndex
    opp_idx = 1 - your_index
    opp_ps = state.players[opp_idx]
    your_ps = state.players[your_index]

    # Guess opponent's facedown active (only needed if active is None)
    opp_active = opp_ps.active
    opp_active_ids = []
    if opp_active and opp_active[0] is None:
        opp_active_ids = [677]  # Riolu — safe Basic Pokémon fallback

    # Determinization: sample from our own deck for your_deck/prize;
    # use a deck with at least one Basic for opponent (mirror match assumption)
    your_deck_sample = random.sample(your_deck, min(your_ps.deckCount, len(your_deck)))
    your_prize_count = sum(1 for p in your_ps.prize if p is not None)
    if your_prize_count == 0:
        your_prize_count = len(your_ps.prize)
    your_prize_sample = random.sample(your_deck, min(your_prize_count, len(your_deck)))

    opp_deck_count = opp_ps.deckCount
    # Use our own deck as proxy for opponent's deck (self-play assumption)
    opp_deck_sample = random.sample(your_deck, min(opp_deck_count, len(your_deck))) if opp_deck_count > 0 else [677]
    opp_prize_count = sum(1 for p in opp_ps.prize if p is not None)
    if opp_prize_count == 0:
        opp_prize_count = len(opp_ps.prize)
    opp_prize_sample = random.sample(your_deck, min(opp_prize_count, len(your_deck)))
    opp_hand_sample = random.sample(your_deck, min(opp_ps.handCount, len(your_deck)))

    try:
        search_state = search_begin(
            obs_class,
            your_deck_sample,
            your_prize_sample,
            opp_deck_sample,
            opp_prize_sample,
            opp_hand_sample,
            opp_active_ids,
        )
    except Exception:
        # Fall back to raw policy if search_begin fails
        actions = enumerate_actions(obs_dict)
        if not actions:
            return [0], None
        _, _, probs, sample = _eval_obs(obs_dict, your_index, your_deck, model, device)
        if sample is None:
            return [0], None
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        sample.action_idx = best_idx
        return actions[best_idx], sample

    # Build root node from search_state
    root_obs = _obs_class_to_dict(search_state.observation)
    root_v, root_actions, root_probs, root_sample = _eval_obs(root_obs, your_index, your_deck, model, device)

    if not root_actions or root_sample is None:
        search_end()
        return [0], None

    root = _Node(value=root_v, total=root_v, visit=1, children=[
        _Child(action=a, prob=p, node=None, search_id=search_state.searchId)
        for a, p in zip(root_actions, root_probs)
    ])

    try:
        # MCTS simulations
        for _ in range(search_count):
            node = root
            path: list[tuple] = []  # (parent_node, child)

            # Selection + expansion
            while True:
                if not node.children:
                    v = node.value
                    for parent, _ in path:
                        parent.total += v
                        parent.visit += 1
                    break
                best_child = max(
                    node.children,
                    key=lambda c: (
                        (c.node.total / c.node.visit if c.node else 0.0)
                        + PUCT_C * math.sqrt(node.visit) * c.prob / (1 + (c.node.visit if c.node else 0))
                    )
                )
                if best_child.node is None:
                    # Expand
                    try:
                        next_state = search_step(best_child.search_id, best_child.action)
                        next_obs = _obs_class_to_dict(next_state.observation)
                        child_v, child_actions, child_probs, _ = _eval_obs(
                            next_obs, your_index, your_deck, model, device
                        )
                        child_node = _Node(value=child_v, total=child_v, visit=1,
                                           children=[
                                               _Child(action=a, prob=p, node=None,
                                                      search_id=next_state.searchId)
                                               for a, p in zip(child_actions, child_probs)
                                           ])
                        best_child.node = child_node
                        v = child_v
                    except Exception:
                        v = 0.0
                    # Backprop
                    for parent, _ in path:
                        parent.total += v
                        parent.visit += 1
                    break
                else:
                    path.append((node, best_child))
                    node = best_child.node

        # Select most-visited child
        best = max(root.children, key=lambda c: c.node.visit if c.node else 0)
        best_idx = root.children.index(best)

        total_visits = sum(c.node.visit for c in root.children if c.node) or 1
        root_sample.action_idx = best_idx
        root_sample.mcts_policy = [
            (c.node.visit / total_visits if c.node else 0.0) for c in root.children
        ]
        root_sample.td_value = root.total / root.visit

        return best.action, root_sample
    finally:
        search_end()


def apply_td_lambda(
    samples: list[LearnSample],
    result: int,
    your_index: int,
    lam: float = 0.9,
) -> None:
    """Update td_value in-place with TD(λ) backward pass."""
    terminal = 1.0 if result == your_index else (-1.0 if result != 2 else 0.0)
    value = terminal
    for sample in reversed(samples):
        value = lam * value + (1.0 - lam) * sample.td_value
        sample.td_value = value


def train_step(
    batch: list[LearnSample],
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One gradient step on a batch. Returns loss as float."""
    model.train()
    losses = []

    for sample in batch:
        board = torch.tensor(sample.board, device=device).unsqueeze(0)
        hand_t = _pad(sample.hand_ids, MAX_PAD_HAND).unsqueeze(0).to(device)
        discard_t = _pad(sample.discard_ids, MAX_PAD_DISCARD).unsqueeze(0).to(device)
        deck_t = _pad(sample.deck_ids, MAX_PAD_DECK).unsqueeze(0).to(device)
        scalars = torch.tensor(sample.scalars, device=device).unsqueeze(0)
        opt_types = torch.tensor(sample.opt_types, dtype=torch.long, device=device)
        opt_cards = torch.tensor(sample.opt_cards, dtype=torch.long, device=device)

        value, scores = model(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)

        v_target = torch.tensor([[sample.td_value]], dtype=torch.float32, device=device)
        loss_v = F.huber_loss(value, v_target, delta=0.2)

        policy_target = torch.tensor(sample.mcts_policy, dtype=torch.float32, device=device)
        log_probs = F.log_softmax(scores, dim=0)
        loss_p = -(policy_target * log_probs).sum()

        losses.append(loss_v + loss_p)

    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss.item()


def self_play_game(
    deck: list[int],
    model: PolicyValueNet,
    device: torch.device,
    epsilon: float = 0.1,
    search_count: int = 10,
) -> list[LearnSample]:
    """Play one self-play game. Returns LearnSamples with td_value filled."""
    env = PTCGEnv()
    obs = env.reset(deck, deck, your_index=0)
    samples_by_player: list[list[LearnSample]] = [[], []]
    done = False

    while not done:
        sel = obs.get('select')
        if sel is None:
            obs, done, _ = env.step([])
            continue

        your_index = obs['current']['yourIndex']

        if random.random() < epsilon:
            n_opts = len(sel['option'])
            max_count = sel['maxCount']
            action = random.sample(range(n_opts), min(max_count, n_opts))
            # Still evaluate the position with the model so we collect training data.
            _, _, probs, sample = _eval_obs(obs, your_index, deck, model, device)
            if sample is not None:
                sample.action_idx = action[0] if action else 0
                if not probs:
                    sample = None
        else:
            action, sample = mcts_step(obs, deck, model, device, search_count)

        obs, done, _ = env.step(action)
        if sample is not None:
            samples_by_player[your_index].append(sample)

    env.close()
    result = obs['current']['result']

    all_samples = []
    for pi in range(2):
        apply_td_lambda(samples_by_player[pi], result, your_index=pi)
        all_samples.extend(samples_by_player[pi])
    return all_samples


def eval_vs_random(
    deck: list[int],
    model: PolicyValueNet,
    device: torch.device,
    n_games: int = 50,
) -> float:
    """Return win rate of model vs random opponent."""
    wins = 0
    model.eval()
    for i in range(n_games):
        your_index = i % 2
        env = PTCGEnv()
        obs = env.reset(deck, deck, your_index=your_index)
        done = False
        while not done:
            sel = obs.get('select')
            if sel is None:
                obs, done, _ = env.step([])
                continue
            acting = obs['current']['yourIndex']
            if acting == your_index:
                with torch.no_grad():
                    action, _ = mcts_step(obs, deck, model, device, search_count=3)
            else:
                n_opts = len(sel['option'])
                max_count = sel['maxCount']
                action = random.sample(range(n_opts), min(max_count, n_opts))
            obs, done, _ = env.step(action)
        env.close()
        result = obs['current']['result']
        if result == your_index:
            wins += 1
    model.train()
    return wins / n_games


def run_dmc(
    deck: list[int],
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_path: str = 'model.pt',
    n_iterations: int = 200,
    games_per_iter: int = 10,
    batch_size: int = 32,
    buffer_capacity: int = 50_000,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.05,
    search_count: int = 10,
    eval_every: int = 20,
    eval_games: int = 50,
    gate_winrate: float = 0.80,
) -> None:
    """Outer DMC training loop. Saves model when gate_winrate is reached."""
    buf = RingBuffer(capacity=buffer_capacity)

    for iteration in range(n_iterations):
        eps = epsilon_start + (epsilon_end - epsilon_start) * iteration / n_iterations
        model.eval()
        for _ in range(games_per_iter):
            samples = self_play_game(deck, model, device, epsilon=eps, search_count=search_count)
            for s in samples:
                buf.push(s)

        if len(buf) >= batch_size:
            batch = buf.sample(batch_size)
            loss = train_step(batch, model, optimizer, device)
            print(f"iter {iteration:4d} | buf {len(buf):6d} | loss {loss:.4f} | eps {eps:.3f}")

        if (iteration + 1) % eval_every == 0:
            wr = eval_vs_random(deck, model, device, n_games=eval_games)
            print(f"  >> eval win rate vs random: {wr:.1%}")
            torch.save({'model': model.state_dict(), 'iteration': iteration}, save_path)
            if wr >= gate_winrate:
                print(f"  Gate passed ({wr:.1%} >= {gate_winrate:.1%}) — Phase 1 complete!")
                return

    print("Training finished.")
