# search/ismcts.py
import math
import random
import time
from dataclasses import dataclass, field

import torch

from cg.api import search_begin, search_step, search_end, to_observation_class
from core.belief import BeliefState
from core.features import enumerate_actions
from model.net import PolicyValueNet
from train.buffer import LearnSample
from train.dmc import _eval_obs, _obs_class_to_dict, PUCT_C

K_DETERMINIZATIONS = 3


@dataclass
class _Child:
    action: list[int]
    prob: float
    node: '_Node | None' = None
    search_id: int = 0


@dataclass
class _Node:
    value: float
    total: float
    visit: int
    children: list[_Child] = field(default_factory=list)


def _run_one_puct(
    obs_class, your_deck, your_deck_sample, your_prize_full,
    opp_deck_sample, opp_prize_full, opp_hand_sample, opp_active_ids,
    your_index, model, device, search_count, deadline,
) -> list[float]:
    """Run one PUCT tree for one determinization. Returns visit proportions per root action."""
    try:
        search_state = search_begin(obs_class, your_deck_sample, your_prize_full,
                                     opp_deck_sample, opp_prize_full, opp_hand_sample, opp_active_ids)
    except Exception:
        return []

    root_obs = _obs_class_to_dict(search_state.observation)
    root_v, root_actions, root_probs, _ = _eval_obs(root_obs, your_index, your_deck, model, device)
    if not root_actions:
        search_end()
        return []

    root = _Node(value=root_v, total=root_v, visit=1, children=[
        _Child(action=a, prob=p, node=None, search_id=search_state.searchId)
        for a, p in zip(root_actions, root_probs)
    ])

    try:
        for _ in range(search_count):
            if time.time() > deadline:
                break
            node = root
            path = []
            while True:
                if not node.children:
                    v = node.value
                    for pn, _ in path:
                        pn.total += v; pn.visit += 1
                    break
                best_child = max(node.children, key=lambda c: (
                    (c.node.total / c.node.visit if c.node else 0.0)
                    + PUCT_C * math.sqrt(node.visit) * c.prob / (1 + (c.node.visit if c.node else 0))
                ))
                if best_child.node is None:
                    try:
                        ns = search_step(best_child.search_id, best_child.action)
                        no = _obs_class_to_dict(ns.observation)
                        cv, ca, cp, _ = _eval_obs(no, your_index, your_deck, model, device)
                        best_child.node = _Node(value=cv, total=cv, visit=1, children=[
                            _Child(action=a, prob=p, node=None, search_id=ns.searchId)
                            for a, p in zip(ca, cp)
                        ])
                        v = cv
                    except Exception:
                        v = 0.0
                    path.append((node, best_child))
                    for pn, _ in path:
                        pn.total += v; pn.visit += 1
                    break
                path.append((node, best_child))
                node = best_child.node
    finally:
        search_end()

    if not root.children:
        return []
    total = sum(c.node.visit if c.node else 0 for c in root.children)
    if total == 0:
        return [1.0 / len(root.children)] * len(root.children)
    return [(c.node.visit if c.node else 0) / total for c in root.children]


def ismcts_step(
    obs_dict: dict,
    your_deck: list[int],
    model: PolicyValueNet,
    device: torch.device,
    belief: BeliefState,
    k: int = K_DETERMINIZATIONS,
    search_count: int = 5,
    timeout_secs: float = 3.0,
) -> tuple[list[int], LearnSample | None]:
    """K-determinization ISMCTS. Pools root visit counts across k determinizations."""
    t0 = time.time()
    deadline = t0 + timeout_secs * 0.85

    obs_class = to_observation_class(obs_dict)
    if obs_class.current is None or obs_class.select is None:
        return [], None

    your_index = obs_class.current.yourIndex
    opp_idx = 1 - your_index
    opp_ps = obs_class.current.players[opp_idx]
    your_ps = obs_class.current.players[your_index]

    belief.update_from_obs(obs_class, opp_idx)

    root_actions = enumerate_actions(obs_dict)
    if not root_actions:
        return [0], None

    n_actions = len(root_actions)
    pooled_visits = [0.0] * n_actions

    for _ in range(k):
        if time.time() > deadline:
            break
        hand_count = opp_ps.handCount
        deck_count = opp_ps.deckCount
        prize_facedown = sum(1 for p in opp_ps.prize if p is None)
        opp_hand, opp_deck, opp_prize_s = belief.sample_determinization(hand_count, deck_count, prize_facedown)
        opp_prize_full = opp_prize_s + [p.id for p in opp_ps.prize if p is not None]

        your_deck_s = random.sample(your_deck, min(your_ps.deckCount, len(your_deck)))
        your_prize_fd = sum(1 for p in your_ps.prize if p is None)
        your_prize_s = random.sample(your_deck, min(your_prize_fd, len(your_deck)))
        your_prize_full = your_prize_s + [p.id for p in your_ps.prize if p is not None]

        opp_active_ids = []
        if opp_ps.active and opp_ps.active[0] is None:
            opp_active_ids = [opp_deck[0] if opp_deck else 677]

        visits = _run_one_puct(obs_class, your_deck, your_deck_s, your_prize_full,
                                opp_deck, opp_prize_full, opp_hand, opp_active_ids,
                                your_index, model, device, search_count, deadline)
        if len(visits) == n_actions:
            for i, v in enumerate(visits):
                pooled_visits[i] += v

    if max(pooled_visits, default=0.0) == 0.0:
        from train.dmc import mcts_step
        return mcts_step(obs_dict, your_deck, model, device, search_count=3)

    best_idx = max(range(n_actions), key=lambda i: pooled_visits[i])
    action = root_actions[best_idx]

    try:
        _, _, probs, sample = _eval_obs(obs_dict, your_index, your_deck, model, device)
        if sample is not None:
            import math as _m
            sample.action_idx = best_idx
            total = sum(pooled_visits)
            sample.mcts_policy = [v / total for v in pooled_visits] if total > 0 else pooled_visits
            sample.log_prob_old = _m.log(max(pooled_visits[best_idx] / total if total > 0 else 1e-8, 1e-8))
    except Exception:
        sample = None

    return action, sample
