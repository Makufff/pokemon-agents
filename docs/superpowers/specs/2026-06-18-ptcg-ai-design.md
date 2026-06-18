# PTCG AI Hyper Solution — Design Spec

**Date:** 2026-06-18
**Goal:** #1 on Kaggle PTCG AI Battle Challenge Simulation ladder + strong Strategy report
**Compute:** Single consumer GPU (RTX 3080/4080 class)
**Timeline:** Competition ends 2026-08-16 (submission lock), leaderboard final ~2026-08-31

---

## Summary

Five-phase end-to-end system targeting #1 on the PTCG ladder. Fresh codebase (not extending the official sample). Core design decisions:

- **Model:** ResNet (positional board) + Deep Sets (unordered hand/bench) — best throughput/expressiveness balance on single GPU
- **Training:** DMC bootstrap → PPO+UPGO league with R-NaD KL-anchoring (anti-cycling) + oracle guiding
- **Inference:** Light ISMCTS using SDK Search API + card-counting belief model
- **Deck:** Fixed Mega Lucario ex through Phase 1–2; evolutionary deck search + Nash mixture from Phase 4
- **Submission:** CPU-safe, fp16-quantized, wall-clock-budgeted

---

## Section 1: Overall Architecture

Four layers build on each other:

```
LAYER 4: Submission
  main.py → ISMCTS (Search API) + Belief Model
            → policy net prior (PUCT) + card-counting

LAYER 3: Trained Models
  Phase 1: DMC model (fast bootstrap)
  Phase 2: PPO+UPGO league model (init from DMC weights)
            + R-NaD / KL-anchor to frozen teacher

LAYER 2: Training Pipeline
  Self-play loop → experience buffer → batch training
  League: main agent + exploiter agent (anti-cycling)

LAYER 1: Core
  env wrapper (Gym-style around libcg.so)
  feature encoder (ResNet board + Deep Sets hand/bench)
  action masking + auto-regressive action heads
```

**Codebase layout:**
```
pokemon-agent/
├── core/
│   ├── env.py          # Gym-style wrapper around libcg.so
│   ├── features.py     # Board/hand/bench feature encoding
│   └── belief.py       # Card-counting belief model
├── model/
│   ├── net.py          # ResNet + Deep Sets policy/value net
│   └── heads.py        # Auto-regressive action heads
├── train/
│   ├── dmc.py          # Phase 1: Deep Monte-Carlo self-play
│   ├── ppo.py          # Phase 2: PPO+UPGO league training
│   └── league.py       # Main + exploiter agent management
├── search/
│   └── ismcts.py       # Inference-time ISMCTS using Search API
├── deck/
│   └── optimize.py     # Phase 4: matchup matrix + Nash mixture
├── main.py             # Submission entry point
└── deck.csv            # Fixed: Mega Lucario ex
```

---

## Section 2: Feature Encoding & Neural Network

### State Encoding

**ResNet branch (positional board):**
- 12 positional slots: active + 5 bench × 2 players
- Each slot: ~40-float vector (normalized HP, damage counters, energy counts per type ×12, special conditions ×5, evolution stage, ex/tera/mega flags, retreat cost)
- Input tensor: 12×40 → 6-layer ResNet, 128 channels → 256-dim board embedding

**Deep Sets branch (unordered sets):**
- Three sets: hand cards, discard pile, deck composition
- Per-card: learned embedding (card ID → 64-dim)
- Shared φ-network (2-layer MLP): 64 → 128-dim per card
- Mean-pool per set → three 128-dim vectors → concatenate → 384-dim set embedding

**Combined:**
- Board (256) + sets (384) + turn scalars (~8) → 648-dim
- 2-layer MLP → value head (scalar, tanh) + policy head

**Turn-level global scalars (concatenated before MLP):**
turn number, first-player flag, supporter-played, energy-attached, prize count difference, opponent prize count, deck size ratio

### Action Heads

PTCG actions are compound (type → target). Two auto-regressive heads:
1. **Type head:** scores over legal option types (PLAY/ATTACK/RETREAT/END/etc.)
2. **Target head:** conditioned on chosen type, scores over card/target options

Both apply the legal-move mask from `obs.select.option` before softmax — illegal moves are structurally impossible.

---

## Section 3: Training Pipeline

### Phase 1 — DMC Bootstrap (Weeks 1–3)

- Single-process self-play loop via `battle_start`/`battle_select`/`battle_finish`
- Both sides use current model with ε-greedy exploration
- **Rewards:** +0.1 per prize taken, −0.1 per prize given, +1.0/−1.0 win/loss; anneals to pure win/loss by week 3
- **TD(λ):** λ=0.9, computed backwards from game end
- Ring buffer: 50k samples; batch size 256
- Frozen teacher snapshot saved every 500 games for Phase 2 KL-anchoring

**Gate:** Must reach >80% win rate vs random opponent before advancing to Phase 2.

### Phase 2 — PPO + UPGO League Training (Weeks 3–8)

**Two concurrent agents:**
- **Main agent:** PPO + UPGO loss + KL-divergence penalty toward frozen DMC teacher (weight 0.01, annealed to 0 by week 6)
- **Exploiter agent:** Trained purely to beat current main agent snapshot (updated every 1000 games). If exploiter win rate vs main exceeds 60%, main agent's KL penalty weight doubles temporarily.

**Oracle guiding (Suphx trick):**
During training only, the value network receives the opponent's full hand as an extra input. At inference this input is zeroed. Speeds up credit assignment without leaking information at test time.

**Self-play matchmaking:**
- Main vs. main: 70%
- Main vs. exploiter: 20%
- Main vs. frozen DMC teacher: 10% (prevents catastrophic forgetting)

---

## Section 4: Inference-Time ISMCTS + Belief Model

### Belief Model (`core/belief.py`)

Maintains a probability distribution over card IDs for each hidden opponent slot, updated from the public log every turn:
- Cards seen played, discarded, or in active/bench are removed from the possible pool
- Deck size decrements tracked per draw; hand size tracked per log event
- Prior at game start: full Standard pool weighted by archetype frequency from the 15k sample self-play win rates

### ISMCTS (`search/ismcts.py`)

Uses SDK `search_begin` / `search_step` / `search_end` / `search_release` — no game rule reimplementation needed.

Per decision step:
1. Sample K=8 determinizations from belief distribution
2. For each: call `search_begin()`, run PUCT tree search with policy net as prior and value net at leaves
3. Information-set node pooling: aggregate visit statistics across determinizations sharing the same visible action history
4. Select action with highest pooled visit count

**Wall-clock budget:** 3-second hard timeout per decision. Degrades to raw policy net if time runs short (prevents timeout losses).

**CPU-only fallback:** K=3, depth cap 4 plies if no GPU detected at submission runtime.

---

## Section 5: Deck Meta-Optimization & Submission Strategy

### Phase 4 — Deck Search (Week 3+ ongoing)

1. **Candidate generation:** Mutate Lucario ex base deck via card swaps from EN_Card_Data.csv (4-copy limit, 1 ACE SPEC, ≥1 Basic Pokémon). Maintain archive of top 20 variants by average win rate.
2. **Matchup matrix:** 200-game mini-tournaments between archive decks; trained agent pilots both sides. M[i,j] = win rate of deck i vs deck j.
3. **Nash mixture:** Solve symmetric zero-sum game from M via `scipy.optimize.linprog`. Submit deck with highest Nash support weight.
4. **Meta adaptation:** Re-estimate M daily from ladder replay data. Shift toward best-response only when top-5 ladder opponents show stable, predictable archetype.

### Phase 5 — Submission Hardening (Ongoing)

- Validate every submission in local mirror match before upload
- Track TrueSkill σ: once μ is high, submit daily to reduce σ (leaderboard ranks on μ−kσ)
- Unit-test SDK engine quirks (timing edge cases, coin-flip RNG, deck-out conditions)
- Bundle: `main.py`, `deck.csv`, `cg/` SDK, fp16-quantized model weights only

---

## Key Constraints & Risks

| Risk | Mitigation |
|---|---|
| GPU not available at inference | CPU-safe fallback: K=3 ISMCTS, depth 4, wall-clock budget |
| Strategic cycling in self-play | KL-anchor to frozen DMC teacher + exploiter agent |
| Deck csv/main.py filename assumptions | Confirmed from sample_submission structure |
| Engine rule differences from real TCG | Do not use external meta sources; derive ground truth from engine self-play |
| 5 submissions/day limit | Test fully locally; only submit when local mirror match passes |
| Report required for prize money | Document matchup matrix, Nash deck, belief model, anti-cycling in Strategy report alongside ladder work |

---

## Phase Timeline

| Week | Phase | Deliverable |
|---|---|---|
| 1–2 | 0: Infrastructure | env wrapper, feature encoder, self-play loop, random baseline |
| 2–4 | 1: DMC Bootstrap | Agent beats random >80%; first ladder submission |
| 3–8 | 4: Deck search (background) | Deck archive + matchup matrix running alongside RL |
| 4–8 | 2: PPO+UPGO League | Main + exploiter agents; oracle guiding; KL-anchor |
| 6–10 | 3: ISMCTS + Belief | Full inference-time search integrated into submission |
| 3–16 | 5: Hardening | σ reduction, quirk testing, meta adaptation, report writing |
