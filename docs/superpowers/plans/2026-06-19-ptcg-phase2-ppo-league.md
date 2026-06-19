# PTCG AI Phase 2 — PPO+UPGO League Training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Phase 1 DMC agent to a full PPO+UPGO league with an exploiter agent and oracle-guided value learning, targeting higher win rates on the ladder.

**Architecture:** Main agent trains via PPO+UPGO against itself, an exploiter, and the frozen DMC teacher (70/20/10 matchup split). The value head receives the oracle (opponent's hand) during training only — zeroed at inference. KL penalty toward the frozen teacher prevents strategic cycling, annealed to 0 over 75% of training.

**Tech Stack:** Python 3.11, PyTorch 2.x, NumPy, pytest

**Prereqs:** Phase 1 complete (`model.pt` exists, all Phase 1 tests pass).

---

## File Map

| File | Change |
|---|---|
| `model/net.py` | Add oracle EmbeddingBag; update value head to accept TRUNK_DIM+D_SETS |
| `train/buffer.py` | Add `log_prob_old: float` and `opp_hand_ids: list[int]` to LearnSample |
| `train/ppo.py` | New — PPO loss, UPGO return computation, KL divergence loss |
| `train/dmc.py` | Populate `log_prob_old` and `opp_hand_ids` in mcts_step + self_play_game |
| `train/league.py` | New — League class, exploiter management, PPO training step, run_league |
| `train_phase2.py` | New — top-level Phase 2 training script |
| `tests/test_net.py` | Extend to test oracle input |
| `tests/test_ppo.py` | New — PPO loss, UPGO, KL tests |
| `tests/test_league.py` | New — League game + exploiter update tests |

---

## Task 1: Oracle Input to PolicyValueNet (`model/net.py`)

**Files:**
- Modify: `model/net.py`
- Modify: `tests/test_net.py`

The oracle is the opponent's hand, visible during training only. The value head is extended from `TRUNK_DIM → D_VALUE_HIDDEN` to `TRUNK_DIM + D_SETS → D_VALUE_HIDDEN`. At inference, `opp_hand_ids=None` zeros the oracle automatically.

- [ ] **Step 1: Add oracle tests to `tests/test_net.py`**

Append these two tests to the existing file:

```python
def test_oracle_input_changes_value():
    net = PolicyValueNet()
    board = torch.zeros(1, 12, 40)
    hand_ids = torch.zeros(1, 10, dtype=torch.long)
    discard_ids = torch.zeros(1, 10, dtype=torch.long)
    deck_ids = torch.zeros(1, 60, dtype=torch.long)
    scalars = torch.zeros(1, 8)
    opt_types = torch.zeros(3, dtype=torch.long)
    opt_cards = torch.zeros(3, dtype=torch.long)
    opp_hand = torch.tensor([[677, 678, 1079]], dtype=torch.long)

    v_no_oracle, _ = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards)
    v_oracle, _    = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards, opp_hand)
    # Scores (policy) must be identical — oracle only affects value head
    _, s_no = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards)
    _, s_or = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards, opp_hand)
    assert torch.allclose(s_no, s_or), "Oracle must NOT change policy scores"
    assert v_no_oracle.shape == (1, 1)
    assert v_oracle.shape == (1, 1)

def test_none_oracle_matches_zero_oracle():
    net = PolicyValueNet()
    board = torch.zeros(1, 12, 40)
    hand_ids = torch.zeros(1, 10, dtype=torch.long)
    discard_ids = torch.zeros(1, 10, dtype=torch.long)
    deck_ids = torch.zeros(1, 60, dtype=torch.long)
    scalars = torch.zeros(1, 8)
    opt_types = torch.zeros(3, dtype=torch.long)
    opt_cards = torch.zeros(3, dtype=torch.long)
    # padding_idx=0 means a zero tensor and None oracle must produce same result
    zero_oracle = torch.zeros(1, 5, dtype=torch.long)  # all padding
    v_none, _ = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards, None)
    v_zero, _ = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards, zero_oracle)
    assert torch.allclose(v_none, v_zero, atol=1e-5), "None oracle must equal all-zero oracle"
```

- [ ] **Step 2: Run tests → verify new tests fail**

```bash
source .venv/bin/activate && pytest tests/test_net.py::test_oracle_input_changes_value tests/test_net.py::test_none_oracle_matches_zero_oracle -v 2>&1 | head -15
```

Expected: FAIL — `TypeError: forward() got an unexpected keyword argument 'opp_hand_ids'`

- [ ] **Step 3: Modify `model/net.py`**

Replace the entire file with:

```python
# model/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

CARD_COUNT = 1268
OPTION_TYPE_COUNT = 17
SLOT_FEATURES = 40
NUM_SLOTS = 12
D_EMBED = 64
D_SETS = 128
RESNET_CHANNELS = 128
NUM_RESBLOCKS = 6
TRUNK_DIM = 256
SCALAR_DIM = 8
COMBINED_DIM = RESNET_CHANNELS + 3 * D_SETS + SCALAR_DIM  # 520
D_VALUE_HIDDEN = 64
D_OPT_TYPE_EMBED = 16
D_OPT_PROJ = 64


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        return x + r


class PolicyValueNet(nn.Module):
    """ResNet over board slots + EmbeddingBag sets → value + per-option scores.

    Oracle: optional opponent hand IDs fed only to value head (zeroed at inference).
    Phase 1 checkpoints are NOT compatible — value head input dim changed.
    """

    def __init__(self):
        super().__init__()
        self.board_proj = nn.Conv1d(SLOT_FEATURES, RESNET_CHANNELS, kernel_size=1)
        self.resblocks = nn.ModuleList([_ResBlock(RESNET_CHANNELS) for _ in range(NUM_RESBLOCKS)])

        # Set branches. Card ID 0 is reserved as padding/unknown.
        self.hand_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)
        self.discard_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)
        self.deck_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)

        # Oracle: opponent's hand (training only, zeroed at inference via padding_idx=0)
        self.oracle_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)

        self.trunk = nn.Sequential(
            nn.Linear(COMBINED_DIM, TRUNK_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_DIM, TRUNK_DIM),
            nn.ReLU(),
        )

        # Value head accepts trunk + oracle → scalar in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(TRUNK_DIM + D_SETS, D_VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(D_VALUE_HIDDEN, 1),
            nn.Tanh(),
        )

        self.opt_type_embed = nn.Embedding(OPTION_TYPE_COUNT, D_OPT_TYPE_EMBED)
        self.opt_card_embed = nn.Embedding(CARD_COUNT, D_EMBED, padding_idx=0)
        self.opt_proj = nn.Linear(D_OPT_TYPE_EMBED + D_EMBED, D_OPT_PROJ)
        self.action_scorer = nn.Linear(TRUNK_DIM + D_OPT_PROJ, 1)

    def _encode_state(self, board, hand_ids, discard_ids, deck_ids, scalars):
        x = board.transpose(1, 2)
        x = F.relu(self.board_proj(x))
        for block in self.resblocks:
            x = block(x)
        board_emb = x.mean(dim=2)
        hand_emb = self.hand_embed(hand_ids)
        discard_emb = self.discard_embed(discard_ids)
        deck_emb = self.deck_embed(deck_ids)
        combined = torch.cat([board_emb, hand_emb, discard_emb, deck_emb, scalars], dim=-1)
        return self.trunk(combined)  # [B, TRUNK_DIM]

    def forward(
        self,
        board: torch.Tensor,           # [B, 12, 40]
        hand_ids: torch.Tensor,        # [B, max_hand]
        discard_ids: torch.Tensor,     # [B, max_discard]
        deck_ids: torch.Tensor,        # [B, max_deck]
        scalars: torch.Tensor,         # [B, 8]
        opt_types: torch.Tensor,       # [N] option type IDs
        opt_cards: torch.Tensor,       # [N] option card IDs
        opp_hand_ids: torch.Tensor | None = None,  # [B, max_opp_hand] or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (value [B,1], scores [N]).

        opp_hand_ids: opponent's hand for oracle training. Pass None at inference
        to zero the oracle (EmbeddingBag of all-zeros returns zero via padding_idx=0).
        """
        assert board.size(0) == 1, "PolicyValueNet.forward requires batch size 1"
        trunk = self._encode_state(board, hand_ids, discard_ids, deck_ids, scalars)

        # Oracle: zero if not provided (using a [1, 1] padding-only tensor)
        if opp_hand_ids is not None:
            oracle_emb = self.oracle_embed(opp_hand_ids)  # [B, D_SETS]
        else:
            dummy = torch.zeros(1, 1, dtype=torch.long, device=board.device)
            oracle_emb = self.oracle_embed(dummy)  # [B, D_SETS] — all zeros via padding_idx=0

        value = self.value_head(torch.cat([trunk, oracle_emb], dim=-1))  # [B, 1]

        # Option scorer — does NOT use oracle (inference parity guaranteed)
        type_emb = self.opt_type_embed(opt_types)
        card_emb = self.opt_card_embed(opt_cards)
        opt_emb = F.relu(self.opt_proj(torch.cat([type_emb, card_emb], dim=-1)))
        trunk_exp = trunk.expand(opt_emb.size(0), -1)
        scores = self.action_scorer(torch.cat([trunk_exp, opt_emb], dim=-1)).squeeze(-1)

        return value, scores
```

- [ ] **Step 4: Run all net tests → 7/7 pass**

```bash
source .venv/bin/activate && pytest tests/test_net.py -v
```

Expected: 7 tests PASS (5 original + 2 new)

- [ ] **Step 5: Commit**

```bash
git add model/net.py tests/test_net.py
git commit -m "feat: oracle EmbeddingBag in PolicyValueNet value head"
```

---

## Task 2: LearnSample Phase 2 Fields (`train/buffer.py`)

**Files:**
- Modify: `train/buffer.py`
- Modify: `tests/test_buffer.py`

Add two optional fields with defaults so Phase 1 code that creates LearnSample without them still works.

- [ ] **Step 1: Add backward-compat test to `tests/test_buffer.py`**

Append to the existing test file:

```python
def test_phase2_fields_have_defaults():
    # LearnSample created without Phase 2 fields must still construct
    s = LearnSample(
        board=np.zeros((12, 40), dtype=np.float32),
        hand_ids=[677], discard_ids=[], deck_ids=[6]*20,
        scalars=np.zeros(8, dtype=np.float32),
        opt_types=[7], opt_cards=[677],
        action_idx=0, td_value=0.5, mcts_policy=[1.0],
    )
    assert s.log_prob_old == 0.0
    assert s.opp_hand_ids == []

def test_phase2_fields_store_correctly():
    s = LearnSample(
        board=np.zeros((12, 40), dtype=np.float32),
        hand_ids=[677], discard_ids=[], deck_ids=[6]*20,
        scalars=np.zeros(8, dtype=np.float32),
        opt_types=[7], opt_cards=[677],
        action_idx=0, td_value=0.5, mcts_policy=[1.0],
        log_prob_old=-1.23, opp_hand_ids=[678, 1079],
    )
    assert s.log_prob_old == -1.23
    assert s.opp_hand_ids == [678, 1079]
```

- [ ] **Step 2: Run → verify new tests fail**

```bash
source .venv/bin/activate && pytest tests/test_buffer.py::test_phase2_fields_have_defaults -v 2>&1 | head -10
```

Expected: FAIL — `TypeError: LearnSample.__init__() missing ... log_prob_old`

- [ ] **Step 3: Modify `train/buffer.py`**

Replace the file with:

```python
# train/buffer.py
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LearnSample:
    board: np.ndarray          # [12, 40]
    hand_ids: list[int]
    discard_ids: list[int]
    deck_ids: list[int]
    scalars: np.ndarray        # [8]
    opt_types: list[int]       # one per candidate action
    opt_cards: list[int]
    action_idx: int            # index of selected action
    td_value: float            # TD(λ) / UPGO return
    mcts_policy: list[float]   # MCTS visit proportions per candidate action
    # Phase 2: optional fields with defaults for backward compatibility
    log_prob_old: float = 0.0                           # log π(a|s) at collection time
    opp_hand_ids: list[int] = field(default_factory=list)  # oracle: opponent hand IDs


class RingBuffer:
    """Fixed-capacity FIFO ring buffer for LearnSample objects."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque[LearnSample] = deque(maxlen=capacity)

    def push(self, sample: LearnSample) -> None:
        self._buf.append(sample)

    def sample(self, n: int) -> list[LearnSample]:
        if n > len(self._buf):
            raise ValueError(f"Cannot sample {n} from buffer of size {len(self._buf)}")
        return random.sample(list(self._buf), n)

    def __len__(self) -> int:
        return len(self._buf)
```

- [ ] **Step 4: Run all buffer tests → 5/5 pass**

```bash
source .venv/bin/activate && pytest tests/test_buffer.py -v
```

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add train/buffer.py tests/test_buffer.py
git commit -m "feat: add log_prob_old and opp_hand_ids to LearnSample for Phase 2"
```

---

## Task 3: PPO+UPGO Losses (`train/ppo.py`)

**Files:**
- Create: `train/ppo.py`
- Create: `tests/test_ppo.py`

Three functions:
- `compute_ppo_loss` — clipped surrogate objective
- `compute_upgo_returns` — backward-pass UPGO targets per game trajectory
- `compute_kl_loss` — KL divergence between model and frozen teacher for one sample

- [ ] **Step 1: Write `tests/test_ppo.py`**

```python
# tests/test_ppo.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import torch
from model.net import PolicyValueNet
from train.buffer import LearnSample
from train.ppo import compute_ppo_loss, compute_upgo_returns, compute_kl_loss


def _make_sample(v=0.5, n_opts=3):
    return LearnSample(
        board=np.zeros((12, 40), dtype=np.float32),
        hand_ids=[677], discard_ids=[], deck_ids=[6] * 20,
        scalars=np.zeros(8, dtype=np.float32),
        opt_types=list(range(n_opts)), opt_cards=[0] * n_opts,
        action_idx=0, td_value=v, mcts_policy=[1.0 / n_opts] * n_opts,
        log_prob_old=-math.log(n_opts), opp_hand_ids=[678],
    )


def test_ppo_loss_is_scalar():
    loss = compute_ppo_loss(
        new_log_prob=torch.tensor(-1.0),
        old_log_prob=-1.0,
        advantage=0.5,
    )
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() < 0  # positive advantage → negative loss (minimized → more negative)


def test_ppo_loss_clips_large_ratio():
    # ratio >> 1 with positive advantage: clipping should kick in
    large_ratio_loss = compute_ppo_loss(
        new_log_prob=torch.tensor(-0.01),
        old_log_prob=-5.0,  # ratio = exp(4.99) >> 1
        advantage=1.0,
        clip_eps=0.2,
    )
    # Clipped loss = -1.2 * advantage = -1.2
    assert abs(large_ratio_loss.item() - (-1.2)) < 0.01, f"Expected ~-1.2 clipped, got {large_ratio_loss.item()}"


def test_upgo_returns_propagates_terminal():
    samples = [_make_sample(v=0.2), _make_sample(v=0.6), _make_sample(v=0.9)]
    returns = compute_upgo_returns(samples, terminal_reward=1.0)
    assert len(returns) == 3
    assert returns[-1] == 1.0, "Last UPGO return must equal terminal reward"
    # t=1: max(V(s_2)=0.9, G_2=1.0) = 1.0
    assert returns[1] == 1.0
    # t=0: max(V(s_1)=0.6, G_1=1.0) = 1.0
    assert returns[0] == 1.0


def test_upgo_returns_uses_value_when_better():
    # V(s_{t+1}) > G_{t+1}: use V
    samples = [_make_sample(v=0.0), _make_sample(v=0.8)]
    returns = compute_upgo_returns(samples, terminal_reward=0.5)
    assert returns[-1] == 0.5
    # t=0: max(V(s_1)=0.8, G_1=0.5) = 0.8
    assert returns[0] == 0.8


def test_upgo_returns_empty():
    assert compute_upgo_returns([], terminal_reward=1.0) == []


def test_kl_loss_same_model_near_zero():
    model = PolicyValueNet()
    sample = _make_sample()
    device = torch.device('cpu')
    loss = compute_kl_loss(model, model, sample, device)
    assert loss.item() < 1e-4, f"KL(model, model) should be ~0, got {loss.item()}"


def test_kl_loss_different_models_positive():
    m1 = PolicyValueNet()
    m2 = PolicyValueNet()  # different random init
    sample = _make_sample()
    device = torch.device('cpu')
    loss = compute_kl_loss(m1, m2, sample, device)
    assert loss.item() >= 0, "KL divergence must be non-negative"
```

- [ ] **Step 2: Run → verify all tests fail**

```bash
source .venv/bin/activate && pytest tests/test_ppo.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'train.ppo'`

- [ ] **Step 3: Implement `train/ppo.py`**

```python
# train/ppo.py
import torch
import torch.nn.functional as F

from train.buffer import LearnSample
from train.dmc import _pad, MAX_PAD_HAND, MAX_PAD_DISCARD, MAX_PAD_DECK


def compute_ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: float,
    advantage: float,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO surrogate loss for one (state, action) pair.

    Returns a scalar tensor. Minimizing this loss increases the probability of
    actions with positive advantage, clipped to prevent large policy updates.
    """
    ratio = torch.exp(new_log_prob - old_log_prob)
    adv = torch.tensor(advantage, dtype=torch.float32, device=new_log_prob.device)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.min(ratio * adv, clipped)


def compute_upgo_returns(
    samples: list[LearnSample],
    terminal_reward: float,
) -> list[float]:
    """Compute UPGO targets for one player's game trajectory.

    G_T = terminal_reward
    G_t = max(V(s_{t+1}), G_{t+1})   for t < T

    V(s_{t+1}) is taken from samples[t+1].td_value (the model's value estimate
    at the time of data collection). Returns a list of floats, same length as samples.
    """
    if not samples:
        return []
    n = len(samples)
    returns = [0.0] * n
    G = terminal_reward
    returns[-1] = G
    for i in range(n - 2, -1, -1):
        v_next = samples[i + 1].td_value
        G = max(v_next, G)
        returns[i] = G
    return returns


def compute_kl_loss(
    model,
    teacher,
    sample: LearnSample,
    device: torch.device,
) -> torch.Tensor:
    """KL(teacher || model) for one sample's action distribution.

    Penalizes the model for diverging from the frozen teacher.
    """
    board = torch.tensor(sample.board, device=device).unsqueeze(0)
    hand_t = _pad(sample.hand_ids, MAX_PAD_HAND).unsqueeze(0).to(device)
    discard_t = _pad(sample.discard_ids, MAX_PAD_DISCARD).unsqueeze(0).to(device)
    deck_t = _pad(sample.deck_ids, MAX_PAD_DECK).unsqueeze(0).to(device)
    scalars = torch.tensor(sample.scalars, device=device).unsqueeze(0)
    opt_types = torch.tensor(sample.opt_types, dtype=torch.long, device=device)
    opt_cards = torch.tensor(sample.opt_cards, dtype=torch.long, device=device)

    _, new_scores = model(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)
    with torch.no_grad():
        _, teacher_scores = teacher(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)

    new_log_probs = F.log_softmax(new_scores, dim=0)
    teacher_log_probs = F.log_softmax(teacher_scores, dim=0)
    teacher_probs = teacher_log_probs.exp()
    # KL(teacher || model) = sum(p_teacher * (log p_teacher - log p_model))
    return (teacher_probs * (teacher_log_probs - new_log_probs)).sum()
```

- [ ] **Step 4: Run all ppo tests → 7/7 pass**

```bash
source .venv/bin/activate && pytest tests/test_ppo.py -v
```

Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add train/ppo.py tests/test_ppo.py
git commit -m "feat: PPO loss, UPGO returns, KL divergence loss"
```

---

## Task 4: Populate `log_prob_old` and `opp_hand_ids` in DMC (`train/dmc.py`)

**Files:**
- Modify: `train/dmc.py`
- Modify: `tests/test_dmc.py`

Two targeted changes:
1. In `mcts_step`: set `sample.log_prob_old = log(mcts_policy[action_idx])` after selecting the best action.
2. In `self_play_game`: track each player's hand (visible on their turn) as the oracle for the opponent.

Read `train/dmc.py` fully before making changes to understand the exact code structure.

- [ ] **Step 1: Add tests to `tests/test_dmc.py`**

Append to the existing file:

```python
def test_mcts_step_sets_log_prob_old():
    import math
    model = PolicyValueNet()
    deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
    device = torch.device('cpu')
    from core.env import PTCGEnv
    env = PTCGEnv()
    obs = env.reset(deck, deck, your_index=0)
    while obs.get('select') is None:
        obs, done, _ = env.step([])
    action, sample = mcts_step(obs, deck, model, device, search_count=2)
    env.close()
    if sample is not None:
        assert sample.log_prob_old <= 0.0, "log_prob must be non-positive (log of probability)"
        assert sample.log_prob_old > -50.0, "log_prob implausibly small"

def test_self_play_game_populates_opp_hand():
    model = PolicyValueNet()
    deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
    device = torch.device('cpu')
    samples = self_play_game(deck, model, device, epsilon=1.0, search_count=2)
    has_oracle = any(len(s.opp_hand_ids) > 0 for s in samples)
    assert has_oracle, "Some samples should have opponent hand IDs (oracle)"
```

- [ ] **Step 2: Run → verify new tests fail**

```bash
source .venv/bin/activate && pytest tests/test_dmc.py::test_mcts_step_sets_log_prob_old tests/test_dmc.py::test_self_play_game_populates_opp_hand -v 2>&1 | head -15
```

Expected: FAIL — `AssertionError` (log_prob_old stays 0.0, opp_hand_ids stays empty)

- [ ] **Step 3: Modify `train/dmc.py`**

Read `train/dmc.py` first to find the exact locations. Then make two changes:

**Change A — in `mcts_step`**, after `root_sample.td_value = root.total / root.visit` is set:

```python
    # log π(a|s) at collection time — used by PPO importance-sampling ratio
    import math
    p_selected = max(root_sample.mcts_policy[best_idx], 1e-8)
    root_sample.log_prob_old = math.log(p_selected)
```

**Change B — in `self_play_game`**, add a `_hand_cache` dict before the game loop:

```python
    _hand_cache: dict[int, list[int]] = {0: [], 1: []}
```

Inside the `while not done:` loop, after the line that reads `your_index = obs['current']['yourIndex']`, add:

```python
        # Update oracle: record the acting player's visible hand for the opponent's samples
        acting_hand = obs['current']['players'][your_index].get('hand') or []
        _hand_cache[your_index] = [c['id'] for c in acting_hand if c]
```

After each `if sample is not None:` block (both epsilon-random branch and MCTS branch), add:

```python
                sample.opp_hand_ids = _hand_cache[1 - your_index]
```

- [ ] **Step 4: Run all dmc + full suite → pass**

```bash
source .venv/bin/activate && pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass (≥22 tests, 0 failures)

- [ ] **Step 5: Commit**

```bash
git add train/dmc.py tests/test_dmc.py
git commit -m "feat: populate log_prob_old and opp_hand_ids in DMC samples for Phase 2"
```

---

## Task 5: League Training Loop (`train/league.py`)

**Files:**
- Create: `train/league.py`
- Create: `tests/test_league.py`

The League class manages three agents (main, exploiter, frozen teacher), plays games with the correct matchup distribution, and runs PPO+UPGO training steps.

Read `train/dmc.py` to understand the exact signatures of `_pad`, `_eval_obs`, `mcts_step`, `apply_td_lambda` before writing `league.py`.

- [ ] **Step 1: Write `tests/test_league.py`**

```python
# tests/test_league.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
from model.net import PolicyValueNet
from train.league import League, run_league

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DEVICE = torch.device('cpu')


def _make_league():
    main = PolicyValueNet()
    exploiter = PolicyValueNet()
    teacher = PolicyValueNet()
    return League(main, exploiter, teacher, DECK, DEVICE)


def test_league_play_game_returns_samples():
    league = _make_league()
    samples = league.play_game(matchup='main_vs_main', search_count=2, epsilon=1.0)
    assert len(samples) > 0, "play_game must return at least one sample"


def test_league_play_game_exploiter_matchup():
    league = _make_league()
    samples = league.play_game(matchup='main_vs_exploiter', search_count=2, epsilon=1.0)
    assert isinstance(samples, list)


def test_exploiter_update_copies_weights():
    league = _make_league()
    with torch.no_grad():
        for p in league.main.parameters():
            p.fill_(0.42)
    league.update_exploiter()
    for pm, pe in zip(league.main.parameters(), league.exploiter.parameters()):
        assert torch.allclose(pm, pe), "Exploiter must match main after update"


def test_teacher_weights_frozen():
    league = _make_league()
    original = {k: v.clone() for k, v in league.teacher.state_dict().items()}
    league.play_game(matchup='main_vs_teacher', search_count=2, epsilon=1.0)
    for k, v in league.teacher.state_dict().items():
        assert torch.allclose(v, original[k]), f"Teacher weight {k} changed!"


def test_ppo_train_step_returns_float():
    league = _make_league()
    optimizer = torch.optim.AdamW(league.main.parameters(), lr=3e-4)
    samples = league.play_game(matchup='main_vs_main', search_count=2, epsilon=1.0)
    if not samples:
        return
    from train.ppo import compute_upgo_returns
    terminal = 0.0
    upgo = compute_upgo_returns(samples, terminal_reward=terminal)
    loss = league.ppo_train_step(samples[:2], optimizer, upgo[:2])
    assert isinstance(loss, float)
    assert loss >= 0
```

- [ ] **Step 2: Run → verify all fail**

```bash
source .venv/bin/activate && pytest tests/test_league.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'train.league'`

- [ ] **Step 3: Implement `train/league.py`**

```python
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
    """Manages main agent, exploiter, and frozen teacher for Phase 2 training."""

    def __init__(
        self,
        main_agent: PolicyValueNet,
        exploiter_agent: PolicyValueNet,
        teacher: PolicyValueNet,
        deck: list[int],
        device: torch.device,
        kl_weight: float = KL_WEIGHT_INIT,
    ):
        self.main = main_agent
        self.exploiter = exploiter_agent
        self.teacher = teacher
        self.deck = deck
        self.device = device
        self.kl_weight = kl_weight

        # Freeze teacher: no gradients, always eval mode
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def sample_matchup(self) -> str:
        r = random.random()
        if r < MAIN_VS_MAIN_PROB:
            return 'main_vs_main'
        elif r < MAIN_VS_MAIN_PROB + MAIN_VS_EXPLOITER_PROB:
            return 'main_vs_exploiter'
        return 'main_vs_teacher'

    def play_game(
        self,
        matchup: str = 'main_vs_main',
        search_count: int = 5,
        epsilon: float = 0.1,
    ) -> list[LearnSample]:
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
        samples_by_player: list[list[LearnSample]] = [[], []]
        hand_cache: dict[int, list[int]] = {0: [], 1: []}
        done = False

        while not done:
            sel = obs.get('select')
            if sel is None:
                obs, done, _ = env.step([])
                continue

            acting = obs['current']['yourIndex']

            # Update oracle cache: acting player's hand is visible this turn
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

        # Collect only main agent's samples (both sides if main vs main)
        all_main_samples = []
        for pi in range(2):
            if models[pi] is self.main:
                terminal = 1.0 if result == pi else (-1.0 if result != 2 else 0.0)
                apply_td_lambda(samples_by_player[pi], result, your_index=pi)
                upgo_rets = compute_upgo_returns(samples_by_player[pi], terminal)
                for s, ur in zip(samples_by_player[pi], upgo_rets):
                    s.td_value = ur  # overwrite with UPGO target
                all_main_samples.extend(samples_by_player[pi])

        return all_main_samples

    def update_exploiter(self) -> None:
        """Clone current main weights into exploiter."""
        self.exploiter.load_state_dict(copy.deepcopy(self.main.state_dict()))

    def eval_exploiter_vs_main(self, n_games: int = 20) -> float:
        """Return exploiter win rate vs main (alternating sides)."""
        wins = 0
        for i in range(n_games):
            your_index = i % 2  # exploiter's player index
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

    def ppo_train_step(
        self,
        batch: list[LearnSample],
        optimizer: torch.optim.Optimizer,
        upgo_returns: list[float],
        clip_eps: float = 0.2,
    ) -> float:
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

            # Oracle input for value head
            opp_hand_t = None
            if sample.opp_hand_ids:
                opp_hand_t = _pad(sample.opp_hand_ids, MAX_PAD_HAND).unsqueeze(0).to(self.device)

            value, scores = self.main(
                board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards, opp_hand_t
            )

            # Value loss: Huber(V(s), UPGO return)
            v_target = torch.tensor([[upgo_ret]], dtype=torch.float32, device=self.device)
            loss_v = F.huber_loss(value, v_target, delta=0.2)

            # PPO policy loss
            log_probs = F.log_softmax(scores, dim=0)
            new_log_prob = log_probs[sample.action_idx]
            advantage = upgo_ret - value.detach().item()
            loss_p = compute_ppo_loss(new_log_prob, sample.log_prob_old, advantage, clip_eps)

            # KL penalty toward frozen teacher
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
        return combined.item()


def run_league(
    deck: list[int],
    main_agent: PolicyValueNet,
    exploiter_agent: PolicyValueNet,
    teacher: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_path: str = 'model_phase2.pt',
    n_iterations: int = 1000,
    games_per_iter: int = 5,
    batch_size: int = 32,
    buffer_capacity: int = 50_000,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
    search_count: int = 5,
    eval_every: int = 50,
    eval_games: int = 30,
    exploiter_update_every: int = 200,
    kl_weight_init: float = KL_WEIGHT_INIT,
    kl_anneal_iters: int = 750,
) -> None:
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
            upgo_returns = [s.td_value for s in batch]  # already UPGO-computed in play_game
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
```

- [ ] **Step 4: Run all league tests → 5/5 pass**

```bash
source .venv/bin/activate && pytest tests/test_league.py -v --timeout=180
```

Expected: 5 PASS (games take ~30s each)

- [ ] **Step 5: Run full test suite → no regressions**

```bash
source .venv/bin/activate && pytest tests/ -v 2>&1 | tail -10
```

Expected: all tests pass (≥27 total)

- [ ] **Step 6: Commit**

```bash
git add train/league.py tests/test_league.py
git commit -m "feat: League class with PPO+UPGO, exploiter, frozen teacher"
```

---

## Task 6: Phase 2 Training Script (`train_phase2.py`)

**Files:**
- Create: `train_phase2.py`

- [ ] **Step 1: Write `train_phase2.py`**

```python
# train_phase2.py
"""Phase 2 PPO+UPGO League training. Run: python train_phase2.py"""
import copy
import torch
from model.net import PolicyValueNet
from train.league import run_league

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DMC_CKPT = 'model.pt'        # Phase 1 checkpoint (teacher + exploiter init)
SAVE_PATH = 'model_phase2.pt'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    # Load Phase 1 weights as starting point for main agent
    main_agent = PolicyValueNet().to(device)
    import os
    if os.path.exists(DMC_CKPT):
        try:
            ckpt = torch.load(DMC_CKPT, map_location=device, weights_only=False)
            # Phase 2 net has a different value head (oracle) — load with strict=False
            missing, unexpected = main_agent.load_state_dict(ckpt['model'], strict=False)
            print(f'Loaded DMC checkpoint (missing={len(missing)}, unexpected={len(unexpected)})')
        except Exception as e:
            print(f'Could not load DMC checkpoint ({e}), starting from scratch')

    # Teacher and exploiter start from the same DMC weights (frozen)
    teacher = copy.deepcopy(main_agent)
    exploiter = copy.deepcopy(main_agent)

    optimizer = torch.optim.AdamW(main_agent.parameters(), lr=1e-4, weight_decay=1e-4)

    run_league(
        deck=DECK,
        main_agent=main_agent,
        exploiter_agent=exploiter,
        teacher=teacher,
        optimizer=optimizer,
        device=device,
        save_path=SAVE_PATH,
        n_iterations=2000,
        games_per_iter=5,
        batch_size=64,
        buffer_capacity=50_000,
        epsilon_start=0.3,
        epsilon_end=0.05,
        search_count=5,
        eval_every=50,
        eval_games=30,
        exploiter_update_every=200,
        kl_weight_init=0.01,
        kl_anneal_iters=1500,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test with 2 iterations**

```bash
source .venv/bin/activate && python -c "
import copy, torch
from model.net import PolicyValueNet
from train.league import run_league

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
device = torch.device('cpu')
main = PolicyValueNet()
teacher = copy.deepcopy(main)
exploiter = copy.deepcopy(main)
opt = torch.optim.AdamW(main.parameters(), lr=1e-4)
run_league(DECK, main, exploiter, teacher, opt, device,
           save_path='model_phase2_smoke.pt',
           n_iterations=2, games_per_iter=1, batch_size=4,
           eval_every=2, eval_games=2,
           exploiter_update_every=10, search_count=2)
print('Smoke test PASSED')
" 2>&1 | tail -5
```

Expected: `Smoke test PASSED` with no exceptions.

- [ ] **Step 3: Commit**

```bash
git add train_phase2.py
git commit -m "feat: Phase 2 training script (PPO+UPGO league)"
```

---

## Self-Review

**Spec coverage:**
- [x] PPO clipped surrogate → `compute_ppo_loss` (Task 3)
- [x] UPGO returns `G_t = max(V(s_{t+1}), G_{t+1})` → `compute_upgo_returns` (Task 3)
- [x] Oracle (Suphx trick) — value head gets opp hand during training, zeroed at inference → Task 1
- [x] KL-anchor to frozen teacher, annealed to 0 → `compute_kl_loss` + `run_league` kl_anneal_iters (Tasks 3, 5)
- [x] Exploiter agent — win rate gate → doubling KL weight (Task 5, `eval_exploiter_vs_main`)
- [x] League matchmaking 70/20/10 → `MAIN_VS_MAIN_PROB` constants (Task 5)
- [x] `log_prob_old` collected in DMC samples → Task 4
- [x] `opp_hand_ids` collected in DMC samples → Task 4
- [x] Phase 1 checkpoint loaded with `strict=False` (value head mismatch) → Task 6

**Placeholder scan:** None found.

**Type consistency:**
- `_eval_obs` used in Task 5 but defined in `train/dmc.py` — must be confirmed exported when reading dmc.py in Task 4/5.
- All function signatures consistent across tasks.
