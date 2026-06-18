# PTCG AI Phase 0+1 — Infrastructure & DMC Bootstrap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build core infrastructure and an MCTS+DMC self-play training loop producing a first competitive ladder submission (>80% win rate vs random opponent).

**Architecture:** `PTCGEnv` wraps `libcg.so`; `core/features.py` encodes the 12-slot board as `[12,40]` + EmbeddingBag mean-pools for hand/discard/deck sets; `PolicyValueNet` uses a 6-layer Conv1d ResNet over board + 3× set embeddings → 520-dim trunk → value head + per-option scorer; MCTS (10 sims, SDK `search_begin/step`) generates training data; TD(λ=0.9) trains both heads from a 50k-sample ring buffer.

**Tech Stack:** Python 3.11, PyTorch 2.x, NumPy, pytest, SciPy

**Scope note:** Covers Phase 0 (Infrastructure) + Phase 1 (DMC Bootstrap) from `docs/superpowers/specs/2026-06-18-ptcg-ai-design.md`. Plans for Phases 2–5 follow after the gate check in Task 9 passes.

**Known constants (from SDK):**
- `card_count = 1268` (max card ID + 1)
- `attack_count = 1557` (max attack ID + 1)
- Mega Lucario ex deck: 4× Riolu (677), 3× Mega Lucario ex (678), trainers, 23× Fighting Energy (6)

---

## File Map

| File | Responsibility |
|---|---|
| `cg/` | SDK — copy of `sample_submission/cg/` (never modify) |
| `core/__init__.py` | empty |
| `core/env.py` | `PTCGEnv`: reset/step/close wrapping SDK battle API |
| `core/features.py` | `encode_board`, `encode_sets`, `encode_scalars`, `enumerate_actions`, `encode_option`, `CARD_TABLE` |
| `model/__init__.py` | empty |
| `model/net.py` | `PolicyValueNet`: 6-layer Conv1d ResNet + EmbeddingBag sets + option scorer |
| `train/__init__.py` | empty |
| `train/buffer.py` | `RingBuffer`: fixed-size ring buffer storing `LearnSample` namedtuples |
| `train/dmc.py` | `mcts_step`, `self_play_game`, `train_step`, `eval_vs_random` |
| `tests/__init__.py` | empty |
| `tests/test_env.py` | env reset/step/error handling |
| `tests/test_features.py` | shape + dtype assertions for each encoding function |
| `tests/test_net.py` | forward-pass shape tests |
| `tests/test_buffer.py` | push/sample/overflow tests |
| `main.py` | submission entry point: loads model, MCTS inference |
| `deck.csv` | 60-line Mega Lucario ex deck |
| `train_phase1.py` | top-level training script + gate check |
| `requirements.txt` | pinned dependencies |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `core/__init__.py`, `model/__init__.py`, `train/__init__.py`, `tests/__init__.py`
- Copy: `cg/` from `sample_submission/cg/`

- [ ] **Step 1: Copy SDK into project root**

```bash
cp -r sample_submission/cg .
```

Expected: `ls cg/` shows `__init__.py api.py game.py sim.py utils.py libcg.so`

- [ ] **Step 2: Create package skeleton**

```bash
mkdir -p core model train tests
touch core/__init__.py model/__init__.py train/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write requirements.txt**

```
torch>=2.2.0
numpy>=1.26
scipy>=1.12
pytest>=8.0
```

- [ ] **Step 4: Verify SDK is importable**

```bash
source .venv/bin/activate
python -c "from cg.api import all_card_data; cards = all_card_data(); print(f'SDK OK: {len(cards)} cards')"
```

Expected output: `SDK OK: 1267 cards` (or similar count)

- [ ] **Step 5: Commit**

```bash
git add cg/ core/__init__.py model/__init__.py train/__init__.py tests/__init__.py requirements.txt
git commit -m "feat: project skeleton and SDK copy"
```

---

## Task 2: PTCGEnv (`core/env.py`)

**Files:**
- Create: `core/env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.env import PTCGEnv

SAMPLE_DECK = [677]*4 + [678]*3 + [1079]*4 + [1086]*4 + [1121]*4 + \
              [1082]*2 + [1097]*2 + [1123]*2 + [1182]*2 + [1210]*4 + \
              [1190]*4 + [1188]*2 + [6]*23

def test_reset_returns_obs_dict():
    env = PTCGEnv()
    obs = env.reset(SAMPLE_DECK, SAMPLE_DECK, your_index=0)
    env.close()
    assert isinstance(obs, dict)
    assert 'current' in obs

def test_game_reaches_done():
    import random
    env = PTCGEnv()
    obs = env.reset(SAMPLE_DECK, SAMPLE_DECK, your_index=0)
    done = False
    steps = 0
    while not done and steps < 5000:
        n_opts = len(obs['select']['option']) if obs.get('select') else 1
        max_count = obs['select']['maxCount'] if obs.get('select') else 1
        action = random.sample(range(n_opts), max_count)
        obs, done, _ = env.step(action)
        steps += 1
    env.close()
    assert done, f"Game did not finish in {steps} steps"

def test_bad_deck_raises():
    import pytest
    env = PTCGEnv()
    bad_deck = [6] * 60  # all energy, no Basic Pokemon
    with pytest.raises(ValueError):
        env.reset(bad_deck, bad_deck)
    env.close()
```

- [ ] **Step 2: Run test → FAIL**

```bash
source .venv/bin/activate && pytest tests/test_env.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'core.env'`

- [ ] **Step 3: Implement `core/env.py`**

```python
# core/env.py
from cg.game import battle_start, battle_finish, battle_select


class PTCGEnv:
    """Gym-like wrapper around libcg.so battle API."""

    _ERROR_MSGS = {
        1: "Invalid card ID in deck",
        2: "More than 4 copies of a named card",
        3: "No Basic Pokémon in deck",
        4: "More than 1 ACE SPEC card",
    }

    def __init__(self):
        self._your_index = 0

    def reset(self, deck0: list[int], deck1: list[int], your_index: int = 0) -> dict:
        """Start a new game. Returns the first observation dict."""
        self._your_index = your_index
        obs, start_data = battle_start(deck0, deck1)
        if start_data.errorPlayer >= 0:
            msg = self._ERROR_MSGS.get(start_data.errorType, "unknown deck error")
            raise ValueError(f"Player {start_data.errorPlayer} deck error: {msg}")
        return obs

    def step(self, action: list[int]) -> tuple[dict, bool, dict]:
        """Apply action indices. Returns (next_obs, done, info).
        
        Reward is NOT computed here — the training loop handles TD(λ) returns.
        info['result'] is set to the winner index (0/1) or 2 for draw when done.
        """
        obs = battle_select(action)
        result = obs['current']['result']
        done = result >= 0
        return obs, done, {'result': result, 'your_index': self._your_index}

    def close(self):
        """Free game memory."""
        battle_finish()
```

- [ ] **Step 4: Run tests → PASS**

```bash
source .venv/bin/activate && pytest tests/test_env.py -v
```

Expected: all 3 tests PASS (the game test may take ~10 seconds)

- [ ] **Step 5: Commit**

```bash
git add core/env.py tests/test_env.py
git commit -m "feat: PTCGEnv wrapping libcg.so battle API"
```

---

## Task 3: Feature Encoding (`core/features.py`)

**Files:**
- Create: `core/features.py`
- Create: `tests/test_features.py`

Board slots: `[your_active, your_bench_0..4, opp_active, opp_bench_0..4]` = 12 slots × 40 features.

Slot feature layout (indices):
- `[0]` is_empty flag
- `[1]` hp / 400
- `[2]` damage_taken / 400 (= (maxHp - hp) / 400)
- `[3]` appeared_this_turn
- `[4–15]` energy count per EnergyType 0–11, each / 10
- `[16]` energy_card_count / 5
- `[17]` tool_count / 3
- `[18]` is_ex (from CARD_TABLE)
- `[19]` is_tera
- `[20]` is_mega_ex
- `[21]` is_basic
- `[22]` is_stage1
- `[23]` is_stage2
- `[24]` retreat_cost / 5
- `[25]` pre_evolution_count / 3
- `[26–39]` zeros (reserved)

Scalar features (8): `[turn/10, is_first_player, supporter_played, energy_attached, your_prizes/6, opp_prizes/6, your_deck/60, opp_deck/60]`

Option encoding: `(option_type_id: int 0–16, card_id: int 0..1267)` per option.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.env import PTCGEnv
from core.features import encode_board, encode_sets, encode_scalars, enumerate_actions, encode_option

SAMPLE_DECK = [677]*4 + [678]*3 + [1079]*4 + [1086]*4 + [1121]*4 + \
              [1082]*2 + [1097]*2 + [1123]*2 + [1182]*2 + [1210]*4 + \
              [1190]*4 + [1188]*2 + [6]*23

def _get_first_real_obs():
    """Get first obs where select is not None (skip deck selection phase)."""
    import random
    env = PTCGEnv()
    obs = env.reset(SAMPLE_DECK, SAMPLE_DECK, your_index=0)
    # The first select may be None (initial deck submission) — skip it
    while obs.get('select') is None or obs['current'] is None:
        obs, done, _ = env.step([])
        if done:
            break
    env.close()
    return obs

def test_encode_board_shape():
    obs = _get_first_real_obs()
    board = encode_board(obs, your_index=0)
    assert board.shape == (12, 40), f"Expected (12, 40), got {board.shape}"
    assert board.dtype == np.float32

def test_encode_sets_returns_lists():
    obs = _get_first_real_obs()
    hand_ids, discard_ids, deck_ids = encode_sets(obs, your_index=0, your_deck=SAMPLE_DECK)
    assert isinstance(hand_ids, list)
    assert isinstance(discard_ids, list)
    assert isinstance(deck_ids, list)
    assert all(isinstance(i, int) for i in hand_ids)

def test_encode_scalars_shape():
    obs = _get_first_real_obs()
    scalars = encode_scalars(obs, your_index=0)
    assert scalars.shape == (8,)
    assert scalars.dtype == np.float32

def test_enumerate_actions_nonempty():
    obs = _get_first_real_obs()
    actions = enumerate_actions(obs)
    assert len(actions) > 0
    assert len(actions) <= 64
    for a in actions:
        assert isinstance(a, list)
        assert all(isinstance(i, int) for i in a)

def test_encode_option_returns_tuple():
    obs = _get_first_real_obs()
    opt = obs['select']['option'][0]
    otype, card_id = encode_option(opt, obs, your_index=0)
    assert isinstance(otype, int)
    assert isinstance(card_id, int)
    assert 0 <= otype <= 16
    assert 0 <= card_id < 1268
```

- [ ] **Step 2: Run test → FAIL**

```bash
source .venv/bin/activate && pytest tests/test_features.py -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'encode_board' from 'core.features'`

- [ ] **Step 3: Implement `core/features.py`**

```python
# core/features.py
import numpy as np
from cg.api import all_card_data, to_observation_class, AreaType, OptionType

CARD_COUNT = 1268  # max cardId + 1
SLOT_FEATURES = 40
MAX_ACTIONS = 64   # max candidate actions enumerated per decision

# Load card metadata once at import time
_all_cards = all_card_data()
CARD_TABLE = {c.cardId: c for c in _all_cards}


def _encode_slot(poke, is_active: bool, ps_special: dict) -> np.ndarray:
    """Encode a single board slot (Pokemon or empty) as a 40-float vector."""
    feat = np.zeros(SLOT_FEATURES, dtype=np.float32)
    if poke is None:
        feat[0] = 1.0  # is_empty
        return feat

    feat[1] = poke['hp'] / 400.0
    max_hp = poke['maxHp']
    feat[2] = (max_hp - poke['hp']) / 400.0
    feat[3] = float(poke.get('appearThisTurn', False))

    for e in poke.get('energies', []):
        idx = 4 + int(e)
        if idx < 16:
            feat[idx] += 0.1

    feat[16] = len(poke.get('energyCards', [])) / 5.0
    feat[17] = len(poke.get('tools', [])) / 3.0

    card = CARD_TABLE.get(poke['id'])
    if card:
        feat[18] = float(card.ex)
        feat[19] = float(card.tera)
        feat[20] = float(card.megaEx)
        feat[21] = float(card.basic)
        feat[22] = float(card.stage1)
        feat[23] = float(card.stage2)
        feat[24] = card.retreatCost / 5.0

    feat[25] = len(poke.get('preEvolution', [])) / 3.0
    return feat


def encode_board(obs: dict, your_index: int) -> np.ndarray:
    """Encode the full board as [12, 40] float32 tensor.
    
    Slot order: your_active, your_bench[0..4], opp_active, opp_bench[0..4]
    """
    state = obs['current']
    board = np.zeros((12, SLOT_FEATURES), dtype=np.float32)
    slot = 0
    for pi_offset in [0, 1]:
        pi = (your_index + pi_offset) % 2
        ps = state['players'][pi]
        # Active slot
        active_list = ps.get('active', [])
        active = active_list[0] if active_list else None
        board[slot] = _encode_slot(active, is_active=True, ps_special=ps)
        slot += 1
        # Bench slots (up to 5)
        bench = ps.get('bench', [])
        for j in range(5):
            poke = bench[j] if j < len(bench) else None
            board[slot] = _encode_slot(poke, is_active=False, ps_special=ps)
            slot += 1
    return board


def encode_sets(obs: dict, your_index: int, your_deck: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return card ID lists for hand, discard pile, and remaining deck.
    
    Uses 0 as a sentinel for 'unknown card' (opponent's hidden hand falls back to this).
    your_deck should be the full 60-card deck list; we slice to deckCount.
    """
    ps = obs['current']['players'][your_index]
    hand = ps.get('hand') or []
    hand_ids = [c['id'] for c in hand]
    discard_ids = [c['id'] for c in ps.get('discard', [])]
    deck_count = ps.get('deckCount', 0)
    deck_ids = your_deck[:deck_count]
    return hand_ids, discard_ids, deck_ids


def encode_scalars(obs: dict, your_index: int) -> np.ndarray:
    """Encode 8 global scalar features as float32 array."""
    state = obs['current']
    opp_index = 1 - your_index
    your_ps = state['players'][your_index]
    opp_ps = state['players'][opp_index]

    scalars = np.array([
        state['turn'] / 10.0,
        float(state.get('firstPlayer', -1) == your_index),
        float(state.get('supporterPlayed', False)),
        float(state.get('energyAttached', False)),
        len(your_ps.get('prize', [])) / 6.0,
        len(opp_ps.get('prize', [])) / 6.0,
        your_ps.get('deckCount', 0) / 60.0,
        opp_ps.get('deckCount', 0) / 60.0,
    ], dtype=np.float32)
    return scalars


def encode_option(opt: dict, obs: dict, your_index: int) -> tuple[int, int]:
    """Return (option_type_id, card_id) for a single option dict.
    
    card_id is 0 when the option has no associated card (END, YES, NO, etc.).
    """
    otype = int(opt['type'])
    card_id = 0

    state = obs['current']
    ps = state['players'][your_index]

    try:
        match otype:
            case 7:  # PLAY — play card from hand
                hand = ps.get('hand') or []
                idx = opt.get('index', 0)
                if idx < len(hand):
                    card_id = hand[idx]['id']
            case 3 | 8 | 9 | 10 | 11:  # CARD, ATTACH, EVOLVE, ABILITY, DISCARD
                area = opt.get('area')
                idx = opt.get('index', 0)
                pi = opt.get('playerIndex', your_index)
                card_id = _card_id_from_area(obs, area, idx, pi, your_index)
            case 13:  # ATTACK — use attackId as proxy card_id (clamped to card range)
                card_id = min(opt.get('attackId', 0), CARD_COUNT - 1)
    except (IndexError, KeyError, TypeError):
        card_id = 0

    return otype, max(0, min(card_id, CARD_COUNT - 1))


def _card_id_from_area(obs: dict, area: int | None, index: int, player_index: int, your_index: int) -> int:
    if area is None:
        return 0
    state = obs['current']
    ps = state['players'][player_index]
    try:
        match area:
            case 2:  # HAND
                hand = ps.get('hand') or []
                return hand[index]['id'] if index < len(hand) else 0
            case 3:  # DISCARD
                discard = ps.get('discard', [])
                return discard[index]['id'] if index < len(discard) else 0
            case 4:  # ACTIVE
                active = ps.get('active', [])
                return active[index]['id'] if index < len(active) else 0
            case 5:  # BENCH
                bench = ps.get('bench', [])
                return bench[index]['id'] if index < len(bench) else 0
            case 6:  # PRIZE
                prize = ps.get('prize', [])
                p = prize[index] if index < len(prize) else None
                return p['id'] if p else 0
            case 7:  # STADIUM
                stadium = state.get('stadium', [])
                return stadium[index]['id'] if index < len(stadium) else 0
            case _:
                return 0
    except (IndexError, KeyError, TypeError):
        return 0


def enumerate_actions(obs: dict) -> list[list[int]]:
    """Return up to MAX_ACTIONS candidate action lists from the current observation.
    
    For maxCount=1, each action is [i] for each legal option index i.
    For maxCount>1, enumerate combinations up to MAX_ACTIONS.
    """
    sel = obs.get('select')
    if not sel:
        return [[]]

    n_opts = len(sel['option'])
    max_count = sel['maxCount']
    min_count = sel['minCount']

    if max_count == 1 or min_count == max_count == 1:
        return [[i] for i in range(n_opts)]

    # Multi-select: enumerate combinations greedily up to MAX_ACTIONS
    actions = []
    indices = list(range(max_count))
    while len(actions) < MAX_ACTIONS:
        if all(i < n_opts for i in indices):
            actions.append(indices.copy())
        # Increment in combinatorial order
        for pos in range(max_count - 1, -1, -1):
            if indices[pos] < n_opts - (max_count - pos):
                indices[pos] += 1
                for j in range(pos + 1, max_count):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break

    return actions if actions else [[0]]
```

- [ ] **Step 4: Run tests → PASS**

```bash
source .venv/bin/activate && pytest tests/test_features.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/features.py tests/test_features.py
git commit -m "feat: board/set/scalar/option feature encoding"
```

---

## Task 4: PolicyValueNet (`model/net.py`)

**Files:**
- Create: `model/net.py`
- Create: `tests/test_net.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_net.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.net import PolicyValueNet, CARD_COUNT

def _dummy_inputs(n_options=5):
    board = torch.zeros(1, 12, 40)
    hand_ids = torch.zeros(1, 10, dtype=torch.long)
    discard_ids = torch.zeros(1, 10, dtype=torch.long)
    deck_ids = torch.zeros(1, 60, dtype=torch.long)
    scalars = torch.zeros(1, 8)
    opt_types = torch.zeros(n_options, dtype=torch.long)
    opt_cards = torch.zeros(n_options, dtype=torch.long)
    return board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards

def test_forward_value_shape():
    net = PolicyValueNet()
    inputs = _dummy_inputs(5)
    value, scores = net(*inputs)
    assert value.shape == (1, 1), f"value shape {value.shape}"
    assert scores.shape == (5,), f"scores shape {scores.shape}"

def test_value_in_range():
    net = PolicyValueNet()
    inputs = _dummy_inputs(3)
    value, _ = net(*inputs)
    assert -1.0 <= value.item() <= 1.0

def test_scores_are_finite():
    net = PolicyValueNet()
    inputs = _dummy_inputs(10)
    _, scores = net(*inputs)
    assert torch.isfinite(scores).all()

def test_zero_options_handled():
    net = PolicyValueNet()
    inputs = _dummy_inputs(1)
    value, scores = net(*inputs)
    assert scores.shape == (1,)

def test_different_option_counts():
    net = PolicyValueNet()
    for n in [1, 5, 20, 64]:
        inputs = _dummy_inputs(n)
        _, scores = net(*inputs)
        assert scores.shape == (n,), f"n={n} scores shape {scores.shape}"
```

- [ ] **Step 2: Run test → FAIL**

```bash
source .venv/bin/activate && pytest tests/test_net.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'model.net'`

- [ ] **Step 3: Implement `model/net.py`**

```python
# model/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

CARD_COUNT = 1268   # max cardId + 1
OPTION_TYPE_COUNT = 17  # OptionType enum 0–16
SLOT_FEATURES = 40
NUM_SLOTS = 12
D_EMBED = 64
D_SETS = 128
RESNET_CHANNELS = 128
NUM_RESBLOCKS = 6
TRUNK_DIM = 256
SCALAR_DIM = 8
# Combined input dim: RESNET_CHANNELS + 3*D_SETS + SCALAR_DIM = 128+384+8 = 520
COMBINED_DIM = RESNET_CHANNELS + 3 * D_SETS + SCALAR_DIM


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        r = x
        x = x.transpose(1, 2)  # [B, L, C]
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [B, C, L]
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        return x + r


class PolicyValueNet(nn.Module):
    """ResNet over board slots + EmbeddingBag sets → value + per-option scores."""

    def __init__(self):
        super().__init__()
        # Board branch: project slot features → ResNet → global avg pool
        self.board_proj = nn.Conv1d(SLOT_FEATURES, RESNET_CHANNELS, kernel_size=1)
        self.resblocks = nn.ModuleList([_ResBlock(RESNET_CHANNELS) for _ in range(NUM_RESBLOCKS)])

        # Set branches (EmbeddingBag with mean pooling; padding_idx=0 ignored)
        self.hand_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)
        self.discard_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)
        self.deck_embed = nn.EmbeddingBag(CARD_COUNT, D_SETS, mode='mean', padding_idx=0)

        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(COMBINED_DIM, TRUNK_DIM),
            nn.ReLU(),
            nn.Linear(TRUNK_DIM, TRUNK_DIM),
            nn.ReLU(),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(TRUNK_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Option embedding (type + card)
        self.opt_type_embed = nn.Embedding(OPTION_TYPE_COUNT, 16)
        self.opt_card_embed = nn.Embedding(CARD_COUNT, D_EMBED, padding_idx=0)
        self.opt_proj = nn.Linear(16 + D_EMBED, 64)

        # Action scorer
        self.action_scorer = nn.Linear(TRUNK_DIM + 64, 1)

    def _encode_state(
        self,
        board: torch.Tensor,         # [B, 12, 40]
        hand_ids: torch.Tensor,      # [B, max_hand]
        discard_ids: torch.Tensor,   # [B, max_discard]
        deck_ids: torch.Tensor,      # [B, max_deck]
        scalars: torch.Tensor,       # [B, 8]
    ) -> torch.Tensor:               # [B, TRUNK_DIM]
        # Board: [B, 40, 12] for Conv1d
        x = board.transpose(1, 2)
        x = F.relu(self.board_proj(x))   # [B, 128, 12]
        for block in self.resblocks:
            x = block(x)
        board_emb = x.mean(dim=2)        # [B, 128]

        hand_emb = self.hand_embed(hand_ids)        # [B, 128]
        discard_emb = self.discard_embed(discard_ids)
        deck_emb = self.deck_embed(deck_ids)

        combined = torch.cat([board_emb, hand_emb, discard_emb, deck_emb, scalars], dim=-1)
        return self.trunk(combined)  # [B, TRUNK_DIM]

    def forward(
        self,
        board: torch.Tensor,         # [B, 12, 40]
        hand_ids: torch.Tensor,      # [B, max_hand]
        discard_ids: torch.Tensor,   # [B, max_discard]
        deck_ids: torch.Tensor,      # [B, max_deck]
        scalars: torch.Tensor,       # [B, 8]
        opt_types: torch.Tensor,     # [N] option type IDs
        opt_cards: torch.Tensor,     # [N] option card IDs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (value [B,1], scores [N])."""
        trunk = self._encode_state(board, hand_ids, discard_ids, deck_ids, scalars)
        value = self.value_head(trunk)  # [B, 1]

        # Option embeddings
        type_emb = self.opt_type_embed(opt_types)   # [N, 16]
        card_emb = self.opt_card_embed(opt_cards)   # [N, D_EMBED]
        opt_emb = F.relu(self.opt_proj(torch.cat([type_emb, card_emb], dim=-1)))  # [N, 64]

        # Score each option against state (B=1 assumed for inference)
        trunk_exp = trunk.expand(opt_emb.size(0), -1)  # [N, TRUNK_DIM]
        scores = self.action_scorer(torch.cat([trunk_exp, opt_emb], dim=-1)).squeeze(-1)  # [N]
        return value, scores
```

- [ ] **Step 4: Run tests → PASS**

```bash
source .venv/bin/activate && pytest tests/test_net.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/net.py tests/test_net.py
git commit -m "feat: PolicyValueNet with ResNet board + EmbeddingBag sets"
```

---

## Task 5: RingBuffer (`train/buffer.py`)

**Files:**
- Create: `train/buffer.py`
- Create: `tests/test_buffer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_buffer.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from train.buffer import RingBuffer, LearnSample

def _make_sample(value=0.5):
    return LearnSample(
        board=np.zeros((12, 40), dtype=np.float32),
        hand_ids=[677, 678],
        discard_ids=[],
        deck_ids=[6] * 20,
        scalars=np.zeros(8, dtype=np.float32),
        opt_types=[7, 14],     # PLAY, END
        opt_cards=[677, 0],
        action_idx=0,
        td_value=value,
        mcts_policy=[0.8, 0.2],
    )

def test_push_and_sample():
    buf = RingBuffer(capacity=100)
    for i in range(10):
        buf.push(_make_sample(float(i)))
    assert len(buf) == 10
    batch = buf.sample(5)
    assert len(batch) == 5

def test_overflow_wraps():
    buf = RingBuffer(capacity=5)
    for i in range(8):
        buf.push(_make_sample(float(i)))
    assert len(buf) == 5  # capped at capacity

def test_cannot_sample_more_than_stored():
    import pytest
    buf = RingBuffer(capacity=100)
    buf.push(_make_sample())
    with pytest.raises(ValueError):
        buf.sample(10)
```

- [ ] **Step 2: Run test → FAIL**

```bash
source .venv/bin/activate && pytest tests/test_buffer.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `train/buffer.py`**

```python
# train/buffer.py
import random
from collections import deque
from dataclasses import dataclass
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
    action_idx: int            # index into opt_types/opt_cards of selected action
    td_value: float            # TD(λ) return
    mcts_policy: list[float]   # MCTS visit proportions per candidate action


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

- [ ] **Step 4: Run tests → PASS**

```bash
source .venv/bin/activate && pytest tests/test_buffer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add train/buffer.py tests/test_buffer.py
git commit -m "feat: RingBuffer for DMC experience storage"
```

---

## Task 6: DMC Self-Play + Training (`train/dmc.py`)

**Files:**
- Create: `train/dmc.py`
- Create: `tests/test_dmc.py`

This is the core training loop. It adapts the sample notebook's MCTS approach to use PolicyValueNet.

Key functions:
- `obs_to_tensors(obs, your_deck)` — convert obs dict to model input tensors
- `mcts_step(obs, your_deck, model, device, search_count)` — run MCTS, return `(selected_action, LearnSample)`
- `self_play_game(deck, model, device, epsilon, search_count)` — play one game, return list of `LearnSample`
- `apply_td_lambda(samples, result, your_index, lam, prize_shape)` — compute TD(λ) returns in-place
- `train_step(batch, model, optimizer, device)` — one gradient step, return scalar loss
- `eval_vs_random(deck, model, device, n_games)` — return win rate
- `run_dmc(deck, model, optimizer, device, config)` — outer training loop

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dmc.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.net import PolicyValueNet
from train.dmc import obs_to_tensors, eval_vs_random, self_play_game, train_step
from train.buffer import RingBuffer

DECK = [677]*4 + [678]*3 + [1079]*4 + [1086]*4 + [1121]*4 + \
       [1082]*2 + [1097]*2 + [1123]*2 + [1182]*2 + [1210]*4 + \
       [1190]*4 + [1188]*2 + [6]*23
DEVICE = torch.device('cpu')

def test_self_play_game_returns_samples():
    model = PolicyValueNet()
    samples = self_play_game(DECK, model, DEVICE, epsilon=1.0, search_count=2)
    assert len(samples) > 0
    # Each sample has a td_value set (non-zero game result)
    assert any(s.td_value != 0 for s in samples)

def test_train_step_returns_scalar_loss():
    model = PolicyValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    samples = self_play_game(DECK, model, DEVICE, epsilon=1.0, search_count=2)
    if not samples:
        return
    loss = train_step(samples[:1], model, optimizer, DEVICE)
    assert isinstance(loss, float)
    assert loss >= 0

def test_eval_vs_random_returns_winrate():
    model = PolicyValueNet()
    wr = eval_vs_random(DECK, model, DEVICE, n_games=2)
    assert 0.0 <= wr <= 1.0
```

- [ ] **Step 2: Run test → FAIL**

```bash
source .venv/bin/activate && pytest tests/test_dmc.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `train/dmc.py`**

```python
# train/dmc.py
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
    """Convert an obs dict to model input tensors. All returned tensors have batch-dim 1."""
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
    sel = obs.get('select', {}) or {}
    options = sel.get('option', [])
    type_ids, card_ids = [], []
    for action in actions:
        if not action:
            type_ids.append(0)
            card_ids.append(0)
            continue
        # Use first option in the action (multi-select: first card chosen)
        idx = action[0]
        if idx < len(options):
            t, c = encode_option(options[idx], obs, your_index)
        else:
            t, c = 0, 0
        type_ids.append(t)
        card_ids.append(c)
    return (
        torch.tensor(type_ids, dtype=torch.long, device=device),
        torch.tensor(card_ids, dtype=torch.long, device=device),
    )


@dataclass
class _Node:
    value: float
    total: float
    visit: int
    children: list  # list of _Child


@dataclass
class _Child:
    action: list[int]
    prob: float
    node: '_Node | None'
    search_id: int


def _create_root(search_state, your_index, your_deck, model, device) -> tuple[_Node, LearnSample | None]:
    obs_dict = search_state.observation.__dict__ if hasattr(search_state.observation, '__dict__') else {}
    # Use the raw dict stored in search_state.observation — convert via json trick
    import json, dataclasses
    obs_raw = json.loads(json.dumps(
        dataclasses.asdict(search_state.observation),
        default=lambda o: o.value if hasattr(o, 'value') else str(o)
    ))

    state = obs_raw.get('current') or {}
    result = state.get('result', -1)

    if result >= 0:
        v = 1.0 if result == your_index else (-1.0 if result != 2 else 0.0)
        node = _Node(value=v, total=v, visit=1, children=[])
        return node, None

    board, hand_t, discard_t, deck_t, scalars, hand_ids, discard_ids, deck_ids = obs_to_tensors(obs_raw, your_deck, device)
    actions = enumerate_actions(obs_raw)
    opt_types, opt_cards = _encode_actions(obs_raw, actions, your_index, device)

    with torch.no_grad():
        value, scores = model(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)

    v = value.item()
    if state.get('yourIndex', your_index) != your_index:
        v = -v

    probs = torch.softmax(scores, dim=0).cpu().tolist()
    children = [_Child(action=a, prob=p, node=None, search_id=search_state.searchId)
                for a, p in zip(actions, probs)]

    node = _Node(value=v, total=v, visit=1, children=children)

    # LearnSample (policy targets filled after MCTS)
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
    return node, sample


def mcts_step(obs_dict: dict, your_deck: list[int], model: PolicyValueNet,
              device: torch.device, search_count: int = 10) -> tuple[list[int], LearnSample | None]:
    """Run MCTS from the current obs, return (selected_action, LearnSample)."""
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    opp_idx = 1 - your_index
    opp_active = state.players[opp_idx].active
    opp_active_ids = [1072] if (opp_active and opp_active[0] is None) else []

    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount),
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
        opponent_deck=[1] * state.players[opp_idx].deckCount,
        opponent_prize=[1] * len(state.players[opp_idx].prize),
        opponent_hand=[1] * state.players[opp_idx].handCount,
        opponent_active=opp_active_ids,
    )

    root, sample = _create_root(search_state, your_index, your_deck, model, device)
    if not root.children or sample is None:
        search_end()
        return [0], None

    # MCTS simulations
    for _ in range(search_count):
        node = root
        path: list[tuple[_Node, _Child]] = []

        while True:
            c_best = max(
                node.children,
                key=lambda c: (
                    (c.node.total / c.node.visit if c.node else node.total / node.visit)
                    + PUCT_C * math.sqrt(node.visit) * c.prob / (1 + (c.node.visit if c.node else 0))
                )
            )
            if c_best.node is None:
                try:
                    next_state = search_step(c_best.search_id, c_best.action)
                    child_node, _ = _create_root(next_state, your_index, your_deck, model, device)
                    c_best.node = child_node
                except Exception:
                    break
                # Backprop
                v = child_node.value
                for parent, child in path:
                    if parent.children[0].search_id != your_index:
                        v = -v
                    parent.total += v
                    parent.visit += 1
                break
            else:
                path.append((node, c_best))
                node = c_best.node
                if node.value in (1.0, -1.0, 0.0) and not node.children:
                    v = node.value
                    for parent, _ in path:
                        parent.total += v
                        parent.visit += 1
                    break

    search_end()

    # Select most-visited child
    best = max(root.children, key=lambda c: c.node.visit if c.node else 0)
    best_idx = root.children.index(best)
    sample.action_idx = best_idx

    # MCTS policy targets (visit proportions)
    total_visits = sum(c.node.visit for c in root.children if c.node)
    sample.mcts_policy = [
        (c.node.visit / total_visits if c.node and total_visits > 0 else 0.0)
        for c in root.children
    ]
    sample.td_value = root.total / root.visit

    return best.action, sample


def apply_td_lambda(
    samples: list[LearnSample],
    result: int,
    your_index: int,
    lam: float = 0.9,
    prize_shape: float = 0.0,  # set to 0.1 for shaped rewards, anneals to 0
) -> None:
    """Update td_value in-place using TD(λ) backwards pass."""
    terminal = 1.0 if result == your_index else (-1.0 if result != 2 else 0.0)
    value = terminal
    for sample in reversed(samples):
        label = (value + sample.td_value) * 0.5
        value = value * lam + sample.td_value * (1.0 - lam)
        sample.td_value = label


def train_step(
    batch: list[LearnSample],
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One gradient step on a batch of LearnSamples. Returns loss as float."""
    model.train()
    total_loss = torch.tensor(0.0, device=device)

    for sample in batch:
        board = torch.tensor(sample.board, device=device).unsqueeze(0)
        hand_t = _pad(sample.hand_ids, MAX_PAD_HAND).unsqueeze(0).to(device)
        discard_t = _pad(sample.discard_ids, MAX_PAD_DISCARD).unsqueeze(0).to(device)
        deck_t = _pad(sample.deck_ids, MAX_PAD_DECK).unsqueeze(0).to(device)
        scalars = torch.tensor(sample.scalars, device=device).unsqueeze(0)
        opt_types = torch.tensor(sample.opt_types, dtype=torch.long, device=device)
        opt_cards = torch.tensor(sample.opt_cards, dtype=torch.long, device=device)

        value, scores = model(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)

        # Value loss: Huber between predicted value and TD(λ) return
        v_target = torch.tensor([[sample.td_value]], dtype=torch.float32, device=device)
        loss_v = F.huber_loss(value, v_target, delta=0.2)

        # Policy loss: cross-entropy between scores and MCTS visit proportions
        policy_target = torch.tensor(sample.mcts_policy, dtype=torch.float32, device=device)
        log_probs = F.log_softmax(scores, dim=0)
        loss_p = -(policy_target * log_probs).sum()

        total_loss = total_loss + loss_v + loss_p

    total_loss = total_loss / len(batch)
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
    """Play one self-play game. Returns all LearnSamples with td_value filled."""
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
            sample = None
        else:
            action, sample = mcts_step(obs, deck, model, device, search_count)

        obs, done, info = env.step(action)
        if sample is not None:
            samples_by_player[your_index].append(sample)

    env.close()
    result = obs['current']['result']

    # Compute TD(λ) returns for both players
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
            acting_player = obs['current']['yourIndex']
            if acting_player == your_index:
                with torch.no_grad():
                    action, _ = mcts_step(obs, deck, model, device, search_count=5)
            else:
                n_opts = len(sel['option'])
                max_count = sel['maxCount']
                action = random.sample(range(n_opts), min(max_count, n_opts))
            obs, done, info = env.step(action)
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
        with torch.no_grad():
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
                print(f"  ✓ Gate passed ({wr:.1%} >= {gate_winrate:.1%}) — Phase 1 complete!")
                return

    print("Training finished (gate not yet reached — continue training or proceed to Phase 2).")
```

- [ ] **Step 4: Run tests → PASS**

```bash
source .venv/bin/activate && pytest tests/test_dmc.py -v
```

Expected: all 3 tests PASS (each may take 15–30s due to game simulation)

- [ ] **Step 5: Commit**

```bash
git add train/dmc.py tests/test_dmc.py
git commit -m "feat: DMC self-play with MCTS, TD(lambda), training step"
```

---

## Task 7: `deck.csv` (Mega Lucario ex)

**Files:**
- Create: `deck.csv`

- [ ] **Step 1: Verify the deck is legal with the SDK**

```bash
source .venv/bin/activate && python -c "
from cg.game import battle_start, battle_finish
deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*2+[1097]*2+[1123]*2+[1182]*2+[1210]*4+[1190]*4+[1188]*2+[6]*23
obs, sd = battle_start(deck, deck)
if sd.errorPlayer >= 0:
    print('DECK ERROR', sd.errorType)
else:
    print('Deck OK — game started')
    battle_finish()
"
```

Expected: `Deck OK — game started`

- [ ] **Step 2: Write `deck.csv`**

```
677
677
677
677
678
678
678
1079
1079
1079
1079
1086
1086
1086
1086
1121
1121
1121
1121
1082
1082
1097
1097
1123
1123
1182
1182
1210
1210
1210
1210
1190
1190
1190
1190
1188
1188
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
6
```

- [ ] **Step 3: Commit**

```bash
git add deck.csv
git commit -m "feat: Mega Lucario ex deck (4x Riolu 677, 3x Mega Lucario ex 678, trainers, 23x Fighting Energy)"
```

---

## Task 8: Submission Entry Point (`main.py`)

**Files:**
- Create: `main.py`

The submission `main.py` must handle two cases:
1. `obs.select is None` → deck selection phase: return the deck list
2. `obs.select is not None` → action selection: use MCTS + model

- [ ] **Step 1: Write `main.py`**

```python
# main.py
import os
import random
import torch

from cg.api import to_observation_class

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pt')
DECK_PATH = os.path.join(os.path.dirname(__file__), 'deck.csv')
KAGGLE_PATH = '/kaggle_simulations/agent/'
SEARCH_COUNT = 5   # MCTS sims per decision (CPU-safe budget)
TIMEOUT_SECS = 3.0

# ── Load deck ────────────────────────────────────────────────────────────────
def _read_deck() -> list[int]:
    path = DECK_PATH if os.path.exists(DECK_PATH) else KAGGLE_PATH + 'deck.csv'
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()][:60]

_DECK = _read_deck()

# ── Load model ───────────────────────────────────────────────────────────────
def _load_model():
    try:
        from model.net import PolicyValueNet
        net = PolicyValueNet()
        ckpt_path = MODEL_PATH if os.path.exists(MODEL_PATH) else KAGGLE_PATH + 'model.pt'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            net.load_state_dict(ckpt['model'])
        net.eval()
        return net
    except Exception:
        return None

_MODEL = _load_model()
_DEVICE = torch.device('cpu')

# ── Agent function ────────────────────────────────────────────────────────────
def agent(obs_dict: dict) -> list[int]:
    """Main agent entry point required by the competition."""
    obs = to_observation_class(obs_dict)

    # Deck selection phase
    if obs.select is None:
        return _DECK

    # Fallback: random selection (used if model not loaded or times out)
    n_opts = len(obs.select.option)
    max_count = obs.select.maxCount
    fallback = random.sample(range(n_opts), min(max_count, n_opts))

    if _MODEL is None:
        return fallback

    try:
        import time
        from train.dmc import mcts_step
        t0 = time.time()
        action, _ = mcts_step(obs_dict, _DECK, _MODEL, _DEVICE, search_count=SEARCH_COUNT)
        elapsed = time.time() - t0
        if elapsed > TIMEOUT_SECS:
            return fallback
        return action
    except Exception:
        return fallback
```

- [ ] **Step 2: Test that main.py imports without error**

```bash
source .venv/bin/activate && python -c "import main; print('main.py OK')"
```

Expected: `main.py OK`

- [ ] **Step 3: Test agent function handles None select (deck phase)**

```bash
source .venv/bin/activate && python -c "
from main import agent
# Simulate initial obs dict with select=None
result = agent({'select': None, 'logs': [], 'current': None})
print(f'Deck returned: {len(result)} cards, first={result[0]}')
assert len(result) == 60
print('OK')
"
```

Expected: `Deck returned: 60 cards, first=677` and `OK`

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: submission main.py with MCTS inference and graceful fallback"
```

---

## Task 9: Training Script + Gate Check (`train_phase1.py`)

**Files:**
- Create: `train_phase1.py`

- [ ] **Step 1: Write `train_phase1.py`**

```python
# train_phase1.py
"""Phase 1 DMC training script. Run: python train_phase1.py"""
import torch
from model.net import PolicyValueNet
from train.dmc import run_dmc

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*2+[1097]*2+[1123]*2+[1182]*2+[1210]*4+[1190]*4+[1188]*2+[6]*23

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    model = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    run_dmc(
        deck=DECK,
        model=model,
        optimizer=optimizer,
        device=device,
        save_path='model.pt',
        n_iterations=500,
        games_per_iter=10,
        batch_size=64,
        buffer_capacity=50_000,
        epsilon_start=0.5,
        epsilon_end=0.05,
        search_count=10,
        eval_every=20,
        eval_games=50,
        gate_winrate=0.80,
    )

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test with 2 iterations**

```bash
source .venv/bin/activate && python -c "
import torch
from model.net import PolicyValueNet
from train.dmc import run_dmc
deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*2+[1097]*2+[1123]*2+[1182]*2+[1210]*4+[1190]*4+[1188]*2+[6]*23
device = torch.device('cpu')
model = PolicyValueNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
run_dmc(deck, model, opt, device, save_path='model_test.pt',
        n_iterations=2, games_per_iter=2, batch_size=4,
        eval_every=2, eval_games=2, gate_winrate=0.80)
print('Smoke test passed')
"
```

Expected: prints iter logs + eval win rate + `Smoke test passed` (no crash)

- [ ] **Step 3: Commit, then start full training in background**

```bash
git add train_phase1.py
git commit -m "feat: Phase 1 DMC training script"
```

Start training:
```bash
source .venv/bin/activate && nohup python train_phase1.py > logs/train_phase1.log 2>&1 &
echo "Training PID: $!"
```

Watch progress:
```bash
tail -f logs/train_phase1.log
```

---

## Task 10: Local Validation + Packaging

**Gate check:** `model.pt` must exist and win rate must be ≥80% before continuing.

- [ ] **Step 1: Confirm gate (run after training)**

```bash
source .venv/bin/activate && python -c "
import torch
from model.net import PolicyValueNet
from train.dmc import eval_vs_random
deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*2+[1097]*2+[1123]*2+[1182]*2+[1210]*4+[1190]*4+[1188]*2+[6]*23
device = torch.device('cpu')
model = PolicyValueNet()
ckpt = torch.load('model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model'])
wr = eval_vs_random(deck, model, device, n_games=100)
print(f'Win rate vs random: {wr:.1%}')
assert wr >= 0.80, f'Gate FAILED: {wr:.1%} < 80%'
print('Gate PASSED — ready to submit')
"
```

- [ ] **Step 2: Local mirror match (submission validation)**

```bash
source .venv/bin/activate && python -c "
from cg.game import battle_start, battle_finish, battle_select
from main import agent

deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*2+[1097]*2+[1123]*2+[1182]*2+[1210]*4+[1190]*4+[1188]*2+[6]*23
obs, sd = battle_start(deck, deck)
assert sd.errorPlayer < 0, 'Deck error'

# Both sides use our agent (mirror match = Kaggle validation)
steps = 0
while True:
    result = agent(obs)
    obs = battle_select(result)
    if obs['current']['result'] >= 0:
        break
    steps += 1
    if steps > 10000:
        raise RuntimeError('Game too long')
battle_finish()
print(f'Mirror match passed in {steps} steps, result={obs[\"current\"][\"result\"]}')
"
```

Expected: `Mirror match passed in N steps, result=0` (or 1 or 2)

- [ ] **Step 3: Quantize model to fp16 to reduce submission size**

```bash
source .venv/bin/activate && python -c "
import torch
from model.net import PolicyValueNet
model = PolicyValueNet()
ckpt = torch.load('model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model'])
model = model.half()  # fp16
torch.save({'model': {k: v for k, v in model.state_dict().items()}}, 'model_fp16.pt')
import os
print(f'Full: {os.path.getsize(\"model.pt\")/1e6:.1f} MB')
print(f'fp16: {os.path.getsize(\"model_fp16.pt\")/1e6:.1f} MB')
"
```

Update `main.py` to load `model_fp16.pt` and cast to float before inference:

```python
# In main.py _load_model(), change ckpt path to model_fp16.pt and add:
net = net.float()   # inference in fp32 even if weights stored as fp16
```

- [ ] **Step 4: Build submission tar.gz**

```bash
tar -czvf submission.tar.gz \
  main.py deck.csv model_fp16.pt \
  cg/ core/ model/ train/ search/
ls -lh submission.tar.gz
```

Expected: file exists, size < 100MB

- [ ] **Step 5: Final commit**

```bash
git add model_fp16.pt
git commit -m "Phase 1 complete: DMC model trained, mirror match passed, submission.tar.gz ready"
```

---

## Summary

After Task 10, you have:
- A working Mega Lucario ex agent beating random >80%
- A valid `submission.tar.gz` ready to upload to Kaggle
- A ring buffer + training loop ready for Phase 2 upgrades

**Next:** Plan B covers Phase 2 (PPO+UPGO League Training) and Phase 3 (ISMCTS + Belief Model), which upgrade the trained base from this plan.
