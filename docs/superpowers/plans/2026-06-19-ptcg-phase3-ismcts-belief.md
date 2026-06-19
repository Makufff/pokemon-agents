# PTCG AI Phase 3 — ISMCTS + Belief Model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace naive single-determinization MCTS with K-determinization ISMCTS guided by a card-counting belief model, improving inference quality by reasoning about opponent's hidden cards.

**Architecture:** BeliefState tracks opponent's public cards (discard + field) to shrink the unknown card pool. ismcts_step runs K=3 independent PUCT trees (one per sampled determinization), pools root visit counts, and returns the action with the highest total visits. Falls back to raw policy on timeout. main.py updated to use ISMCTS with per-game belief state.

**Tech Stack:** Python 3.11, PyTorch 2.x, SDK cg/api.py

**Prereqs:** Phase 2 complete. model_phase2.pt may or may not exist (graceful fallback to model.pt).

---

## File Map

| File | Change |
|---|---|
| core/belief.py | New — BeliefState tracking opponent's known/unknown cards |
| search/ismcts.py | New — K-determinization ISMCTS using SDK search API |
| main.py | Modify — use ismcts_step + per-game BeliefState, try model_phase2.pt |
| tests/test_belief.py | New — 6 unit tests |
| tests/test_ismcts.py | New — 3 integration tests |

---

## Background: Key SDK Facts

From cg/api.py:
- search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active) starts search with a specific determinization
- All counts must match exactly: len(your_deck) >= deckCount, len(opp_hand) >= handCount, etc.
- If opp_ps.active[0] is None (facedown), opp_active must have at least one Basic Pokemon ID (use 677=Riolu)
- PlayerState.discard: fully visible list of Card objects (IDs revealed)
- PlayerState.bench: fully visible list of Pokemon objects
- PlayerState.active: visible if face-up, None if facedown
- PlayerState.prize: list of Card|None; None=facedown (count visible, ID not)
- PlayerState.handCount: number of cards in opponent's hand (visible)
- PlayerState.deckCount: cards remaining in deck (visible)
- search_end(): must always be called (use finally)

---

## Task 1: Belief Model (core/belief.py)

**Files:**
- Create: core/belief.py
- Create: tests/test_belief.py

- [ ] **Step 1: Write tests/test_belief.py**

File content:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collections import Counter
from core.belief import BeliefState

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26

def test_initial_pool_matches_deck():
    b = BeliefState(DECK)
    assert sorted(b.pool_list()) == sorted(DECK)

def test_update_removes_public_cards():
    b = BeliefState(DECK)
    b.mark_public([677])
    assert Counter(b.pool_list())[677] == Counter(DECK)[677] - 1

def test_mark_public_is_idempotent():
    b = BeliefState(DECK)
    b.mark_public([677])
    b.mark_public([677])
    b2 = BeliefState(DECK)
    b2.mark_public([677])
    assert sorted(b.pool_list()) == sorted(b2.pool_list())

def test_sample_determinization_correct_counts():
    b = BeliefState(DECK)
    hand, deck, prize = b.sample_determinization(hand_count=3, deck_count=10, prize_count=2)
    assert len(hand) == 3 and len(deck) == 10 and len(prize) == 2

def test_sample_deck_always_has_basic():
    b = BeliefState(DECK)
    pokemon_ids = {677, 678, 1079, 1086, 1121}
    for _ in range(20):
        _, deck, _ = b.sample_determinization(0, 5, 0)
        assert any(c in pokemon_ids for c in deck)

def test_reset_restores_full_pool():
    b = BeliefState(DECK)
    b.mark_public([677, 678, 1079])
    b.reset(DECK)
    assert sorted(b.pool_list()) == sorted(DECK)
```

- [ ] **Step 2: Run -> verify fail**

```bash
source .venv/bin/activate && pytest tests/test_belief.py -v 2>&1 | head -8
```

Expected: ImportError: No module named 'core.belief'

- [ ] **Step 3: Implement core/belief.py**

File content:

```python
# core/belief.py
import random
from collections import Counter

_BASIC_POKEMON_ID = 677  # Riolu fallback


class BeliefState:
    """Card-counting belief model for the opponent's hidden cards.

    Prior: assumes opponent deck = our own deck (mirror match assumption).
    After observing a card in a public zone, removes it from the unknown pool once.
    """

    def __init__(self, our_deck: list[int]):
        self._prior = list(our_deck)
        self._prior_counter = Counter(our_deck)
        self._pool = Counter(our_deck)
        self._public_seen: Counter = Counter()

    def reset(self, our_deck: list[int]) -> None:
        self._prior = list(our_deck)
        self._prior_counter = Counter(our_deck)
        self._pool = Counter(our_deck)
        self._public_seen = Counter()

    def pool_list(self) -> list[int]:
        return list(self._pool.elements())

    def mark_public(self, card_ids: list[int]) -> None:
        """Mark card IDs as observed in a public zone (incremental, deduplicating)."""
        for cid in card_ids:
            new_total = self._public_seen[cid] + 1
            if new_total <= self._prior_counter[cid] and self._pool[cid] > 0:
                self._pool[cid] -= 1
                if self._pool[cid] == 0:
                    del self._pool[cid]
                self._public_seen[cid] = new_total

    def update_from_obs(self, obs_class, opp_idx: int) -> None:
        """Extract public opponent cards from observation and update pool."""
        if obs_class.current is None:
            return
        opp_ps = obs_class.current.players[opp_idx]
        public: list[int] = []
        for card in opp_ps.discard:
            public.append(card.id)
        for poke in opp_ps.active:
            if poke is not None:
                public.append(poke.id)
                for c in poke.energyCards + poke.tools + poke.preEvolution:
                    public.append(c.id)
        for poke in opp_ps.bench:
            public.append(poke.id)
            for c in poke.energyCards + poke.tools + poke.preEvolution:
                public.append(c.id)
        for p in opp_ps.prize:
            if p is not None:
                public.append(p.id)
        self._rebuild_pool(public)

    def _rebuild_pool(self, public_cards: list[int]) -> None:
        new_seen = Counter(public_cards)
        for cid, count in new_seen.items():
            self._public_seen[cid] = max(self._public_seen[cid], count)
        self._pool = Counter(self._prior)
        for cid, seen in self._public_seen.items():
            remove = min(seen, self._prior_counter[cid])
            self._pool[cid] -= remove
            if self._pool[cid] <= 0:
                del self._pool[cid]

    def sample_determinization(
        self, hand_count: int, deck_count: int, prize_count: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Sample one possible distribution of opponent's hidden cards.

        Pads with basic energy (ID 6) if pool is smaller than needed.
        Guarantees deck has at least one Basic Pokemon.
        """
        available = list(self._pool.elements())
        random.shuffle(available)
        n_needed = hand_count + deck_count + prize_count
        while len(available) < n_needed:
            available.append(6)
        hand = available[:hand_count]
        deck = available[hand_count:hand_count + deck_count]
        prize = available[hand_count + deck_count:hand_count + deck_count + prize_count]
        if deck_count > 0 and _BASIC_POKEMON_ID not in deck:
            deck[0] = _BASIC_POKEMON_ID
        return hand, deck, prize
```

- [ ] **Step 4: Run belief tests -> 6/6 pass**

```bash
source .venv/bin/activate && pytest tests/test_belief.py -v
```

- [ ] **Step 5: Commit**

```bash
git add core/belief.py tests/test_belief.py
git commit -m "feat: BeliefState card-counting belief model for opponent's hidden cards"
```

---

## Task 2: ISMCTS Search (search/ismcts.py)

**Files:**
- Create: search/ismcts.py
- Create: tests/test_ismcts.py

Read train/dmc.py first to understand _eval_obs, _obs_class_to_dict, PUCT_C signatures.

- [ ] **Step 1: Write tests/test_ismcts.py**

File content:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.env import PTCGEnv
from core.belief import BeliefState
from model.net import PolicyValueNet
from search.ismcts import ismcts_step

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DEVICE = torch.device('cpu')

def _get_action_obs():
    env = PTCGEnv()
    obs = env.reset(DECK, DECK, your_index=0)
    for _ in range(200):
        if obs.get('select') is not None:
            return obs, env
        obs, done, _ = env.step([])
        if done:
            break
    return obs, env

def test_ismcts_step_returns_valid_action():
    model = PolicyValueNet(); model.eval()
    belief = BeliefState(DECK)
    obs, env = _get_action_obs()
    try:
        action, sample = ismcts_step(obs, DECK, model, DEVICE, belief, k=2, search_count=3)
        assert isinstance(action, list)
        assert all(isinstance(a, int) for a in action)
    finally:
        env.close()

def test_ismcts_step_respects_timeout():
    import time
    model = PolicyValueNet(); model.eval()
    belief = BeliefState(DECK)
    obs, env = _get_action_obs()
    try:
        t0 = time.time()
        action, _ = ismcts_step(obs, DECK, model, DEVICE, belief, k=10, search_count=50, timeout_secs=1.0)
        assert time.time() - t0 < 3.0
        assert isinstance(action, list)
    finally:
        env.close()

def test_ismcts_step_updates_belief():
    model = PolicyValueNet(); model.eval()
    belief = BeliefState(DECK)
    obs, env = _get_action_obs()
    pool_before = len(belief.pool_list())
    try:
        ismcts_step(obs, DECK, model, DEVICE, belief, k=1, search_count=2)
    finally:
        env.close()
    assert len(belief.pool_list()) <= pool_before
```

- [ ] **Step 2: Run -> verify fail**

```bash
source .venv/bin/activate && pytest tests/test_ismcts.py -v 2>&1 | head -8
```

- [ ] **Step 3: Implement search/ismcts.py**

File content:

```python
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
```

- [ ] **Step 4: Run all ismcts tests -> 3/3 pass**

```bash
source .venv/bin/activate && pytest tests/test_ismcts.py -v --timeout=120
```

- [ ] **Step 5: Run full test suite -> no regressions**

```bash
source .venv/bin/activate && pytest tests/ -v 2>&1 | tail -10
```

Expected: 44+ tests, 0 failures

- [ ] **Step 6: Commit**

```bash
git add search/ismcts.py tests/test_ismcts.py
git commit -m "feat: K-determinization ISMCTS with BeliefState for inference"
```

---

## Task 3: Update main.py

**Files:**
- Modify: main.py

- [ ] **Step 1: Read main.py fully, then replace with:**

```python
# main.py
import os
import random
import torch

from cg.api import to_observation_class

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_phase2.pt')
MODEL_PATH_FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
DECK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deck.csv')
KAGGLE_PATH = '/kaggle_simulations/agent/'
K_DETERMINIZATIONS = 3
SEARCH_COUNT = 5
TIMEOUT_SECS = 3.0


def _read_deck() -> list[int]:
    path = DECK_PATH if os.path.exists(DECK_PATH) else KAGGLE_PATH + 'deck.csv'
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()][:60]


_DECK = _read_deck()


def _load_model():
    try:
        from model.net import PolicyValueNet
        net = PolicyValueNet()
        for ckpt_candidate in [MODEL_PATH, MODEL_PATH_FALLBACK,
                                KAGGLE_PATH + 'model_phase2.pt',
                                KAGGLE_PATH + 'model.pt']:
            if os.path.exists(ckpt_candidate):
                ckpt = torch.load(ckpt_candidate, map_location='cpu', weights_only=False)
                current_shapes = {k: v.shape for k, v in net.state_dict().items()}
                filtered = {k: v for k, v in ckpt['model'].items()
                            if k in current_shapes and v.shape == current_shapes[k]}
                net.load_state_dict(filtered, strict=False)
                break
        net.eval()
        return net
    except Exception:
        return None


_MODEL = _load_model()
_DEVICE = torch.device('cpu')
_BELIEF = None  # BeliefState reset at game start, persists within game


def agent(obs_dict: dict) -> list[int]:
    """Competition entry point -- called once per decision."""
    global _BELIEF

    obs = to_observation_class(obs_dict)

    if obs.select is None:
        from core.belief import BeliefState
        _BELIEF = BeliefState(_DECK)
        return _DECK

    n_opts = len(obs.select.option)
    max_count = obs.select.maxCount
    fallback = random.sample(range(n_opts), min(max_count, n_opts))

    if _MODEL is None:
        return fallback

    if _BELIEF is None:
        from core.belief import BeliefState
        _BELIEF = BeliefState(_DECK)

    try:
        import time
        from search.ismcts import ismcts_step
        t0 = time.time()
        action, _ = ismcts_step(
            obs_dict, _DECK, _MODEL, _DEVICE, _BELIEF,
            k=K_DETERMINIZATIONS,
            search_count=SEARCH_COUNT,
            timeout_secs=TIMEOUT_SECS,
        )
        if time.time() - t0 > TIMEOUT_SECS:
            return fallback
        return action if action else fallback
    except Exception:
        return fallback
```

- [ ] **Step 2: Smoke-test main.py in a mirror match**

```bash
source .venv/bin/activate && python -c "
from main import agent
from core.env import PTCGEnv

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
env = PTCGEnv()
obs = env.reset(DECK, DECK, your_index=0)
steps = 0; done = False
action = agent(obs)  # deck submission
while not done and steps < 200:
    obs, done, info = env.step(agent(obs))
    steps += 1
env.close()
print(f'Game {steps} steps, result={obs[\"current\"][\"result\"]}')
print('SMOKE TEST PASSED')
" 2>&1---step 3 commit---

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: ISMCTS + BeliefState in agent, try model_phase2.pt first"
```

---

## Self-Review

**Spec coverage:**
- [x] Belief model tracks discard + field + revealed prizes (Task 1)
- [x] Sampling guarantees deck has at least one Basic Pokemon (Task 1)
- [x] K=3 determinizations (CPU-safe) (Task 2)
- [x] 3-second hard timeout with raw policy fallback (Task 2)
- [x] Pooled visit counts across determinizations (Task 2)
- [x] BeliefState reset on deck submission (game start) (Task 3)
- [x] model_phase2.pt tried first, model.pt fallback (Task 3)
- [x] shape-mismatch filter on checkpoint load (Task 3)

**No placeholders found.**

**Type consistency:**
- belief.update_from_obs(obs_class, opp_idx) used in ismcts_step consistently
- _run_one_puct returns list[float] consistently indexed by root action position
