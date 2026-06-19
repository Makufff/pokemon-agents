"""Phase 4: Deck Optimization via Nash Mixture.

Strategy:
  1. Mutate the base Lucario ex deck (trainer/energy slots only).
  2. Quick-eval each mutation vs the base deck; keep top K in an archive.
  3. Build an N×N matchup matrix (win rates) via self-play.
  4. Solve the symmetric zero-sum game for Nash equilibrium via LP.
  5. Return the archive deck with the highest Nash support weight.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
from scipy.optimize import linprog

from cg.api import all_card_data, CardData
from core.env import PTCGEnv
from train.dmc import mcts_step

# ── Card constants ────────────────────────────────────────────────────────────
BASE_DECK = (
    [677] * 4 + [678] * 3          # Riolu x4, Mega Lucario ex x3
    + [1079] * 4                    # Rare Candy x4
    + [1086] * 4                    # Buddy-Buddy Poffin x4
    + [1121] * 4                    # Ultra Ball x4
    + [1082] * 1                    # Hyper Aroma (ACE SPEC) x1
    + [1097] * 2                    # Night Stretcher x2
    + [1123] * 2                    # Switch x2
    + [1210] * 4                    # Brock's Scouting x4
    + [1190] * 4                    # Bianca's Devotion x4
    + [1188] * 2                    # Ciphermaniac's Codebreaking x2
    + [6] * 26                      # Basic Fighting Energy x26
)

DECK_SIZE = 60
MAX_COPIES = 4
# IDs that form the Lucario evolution line + ACE SPEC — never mutated
PROTECTED_IDS: frozenset[int] = frozenset({677, 678, 1082})
# cardType values for non-Pokémon, non-Tool cards eligible for swapping
_SWAP_TYPES: frozenset[int] = frozenset({1, 3, 4, 5, 6})  # Item, Supporter, Stadium, BasicEnergy, SpecialEnergy


# ── Internal helpers ──────────────────────────────────────────────────────────

def _card_db() -> dict[int, CardData]:
    return {c.cardId: c for c in all_card_data()}


def _swap_pool(db: dict[int, CardData]) -> list[CardData]:
    """Cards eligible to be added via mutation: non-Pokémon, non-ACE SPEC."""
    return [c for c in db.values()
            if c.cardType in _SWAP_TYPES and not c.aceSpec]


# ── Public API ────────────────────────────────────────────────────────────────

def validate_deck(deck: list[int]) -> tuple[bool, str]:
    """Return (valid, reason). Checks length, copy limits, ACE SPEC limit, Basic Pokémon."""
    if len(deck) != DECK_SIZE:
        return False, f"length {len(deck)} != 60"
    db = _card_db()
    counts = Counter(deck)
    ace_count = 0
    for cid, n in counts.items():
        card = db.get(cid)
        if card is None:
            return False, f"unknown card id {cid}"
        if card.aceSpec:
            ace_count += 1
            if n > 1:
                return False, f"ACE SPEC {card.name} appears {n} times (max 1)"
        elif card.cardType == 5:
            pass  # basic energy: unlimited copies allowed
        elif n > MAX_COPIES:
            return False, f"{card.name} appears {n} times (max 4)"
    if ace_count > 1:
        return False, f"{ace_count} ACE SPEC cards (max 1)"
    has_basic = any(db[cid].basic for cid in counts if cid in db)
    if not has_basic:
        return False, "no Basic Pokémon"
    return True, "ok"


def mutate_deck(
    deck: list[int],
    swap_pool: list[CardData],
    rng: random.Random,
    n_swaps: int = 2,
) -> list[int]:
    """Swap n_swaps non-protected slots with random eligible cards.

    Protected slots: PROTECTED_IDS (Riolu, Mega Lucario ex, Hyper Aroma).
    Respects 4-copy and ACE SPEC limits.
    """
    new_deck = list(deck)
    mutable_slots = [i for i, cid in enumerate(new_deck) if cid not in PROTECTED_IDS]

    for _ in range(n_swaps):
        if not mutable_slots:
            break
        slot = rng.choice(mutable_slots)
        old_cid = new_deck[slot]
        counts = Counter(new_deck)

        # Eligible replacements: not over copy limit, no second ACE SPEC
        eligible = []
        for c in swap_pool:
            if c.cardId == old_cid:
                continue
            if c.aceSpec:
                # Only add if deck currently has 0 ACE SPEC cards after removing old_cid
                remaining_ace = sum(
                    1 for cid, n in counts.items()
                    if cid != old_cid and _card_db().get(cid, None) and _card_db()[cid].aceSpec
                )
                if remaining_ace > 0:
                    continue
                if counts.get(c.cardId, 0) >= 1:
                    continue
            else:
                if c.cardType != 5 and counts.get(c.cardId, 0) >= MAX_COPIES:
                    continue
            eligible.append(c)

        if not eligible:
            continue

        new_deck[slot] = rng.choice(eligible).cardId
        mutable_slots = [i for i, cid in enumerate(new_deck) if cid not in PROTECTED_IDS]

    return new_deck


def _play_one_game(
    deck0: list[int],
    deck1: list[int],
    starting_index: int,
    model,
    device,
    search_count: int,
) -> int:
    """Play one game between deck0 and deck1. Returns winner index (0 or 1)."""
    env = PTCGEnv()
    try:
        obs = env.reset(deck0, deck1, your_index=starting_index)
        done = False
        while not done:
            yi: int = obs['current']['yourIndex']
            player_deck = deck0 if yi == 0 else deck1
            action, _ = mcts_step(obs, player_deck, model, device, search_count)
            obs, done, info = env.step(action)
        return info['result']
    finally:
        env.close()


def matchup_win_rate(
    deck_i: list[int],
    deck_j: list[int],
    model,
    device,
    n_games: int = 20,
    search_count: int = 3,
) -> float:
    """Win rate of deck_i vs deck_j. Alternates starting player for fairness."""
    wins = 0
    for g in range(n_games):
        starting = g % 2
        winner = _play_one_game(deck_i, deck_j, starting, model, device, search_count)
        if winner == 0:
            wins += 1
    return wins / n_games


def build_matchup_matrix(
    archive: list[list[int]],
    model,
    device,
    n_games_per_pair: int = 20,
    search_count: int = 3,
) -> np.ndarray:
    """NxN matrix where M[i,j] = win rate of archive[i] vs archive[j].

    Diagonal is 0.5 by convention. M[i,j] + M[j,i] ≈ 1 (filled by symmetry
    after computing each upper-triangle pair once).
    """
    n = len(archive)
    M = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            wr = matchup_win_rate(
                archive[i], archive[j], model, device, n_games_per_pair, search_count
            )
            M[i, j] = wr
            M[j, i] = 1.0 - wr
            print(f"  matchup [{i}] vs [{j}]: {wr:.3f}")
    return M


def nash_mixture(M: np.ndarray) -> np.ndarray:
    """Find Nash equilibrium for symmetric zero-sum game matrix M.

    Solves: maximize v s.t. M^T p >= v*1, sum(p)=1, p>=0.
    Returns probability distribution over decks (Nash mixed strategy).
    Falls back to uniform if LP fails.
    """
    n = len(M)
    # Variables: [p_0, ..., p_{n-1}, v]
    c = [0.0] * n + [-1.0]                                    # minimize -v
    A_ub = [[-M[i, j] for i in range(n)] + [1.0] for j in range(n)]  # -M^T p + v <= 0
    b_ub = [0.0] * n
    A_eq = [[1.0] * n + [0.0]]                                 # sum(p) = 1
    b_eq = [1.0]
    bounds = [(0.0, None)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    if result.success:
        p = np.maximum(result.x[:n], 0.0)
        total = p.sum()
        if total > 0:
            return p / total
    return np.ones(n) / n


def run_deck_search(
    base_deck: list[int],
    model,
    device,
    n_candidates: int = 20,
    k_archive: int = 8,
    quick_games: int = 10,
    matrix_games: int = 20,
    search_count: int = 3,
    n_swaps: int = 2,
    seed: int = 42,
) -> tuple[list[int], np.ndarray, list[list[int]]]:
    """Full deck search pipeline.

    Returns (best_deck, nash_weights, archive) where best_deck is the archive
    entry with the highest Nash equilibrium support weight.
    """
    rng = random.Random(seed)
    db = _card_db()
    pool = _swap_pool(db)

    print(f"Generating {n_candidates} candidate decks...")
    candidates: list[list[int]] = []
    for _ in range(n_candidates):
        mut = mutate_deck(base_deck, pool, rng, n_swaps=n_swaps)
        ok, reason = validate_deck(mut)
        if ok:
            candidates.append(mut)
        else:
            print(f"  invalid mutation ({reason}), skipping")

    print(f"Quick-evaluating {len(candidates)} candidates vs base deck ({quick_games} games each)...")
    scored: list[tuple[float, list[int]]] = []
    for i, cand in enumerate(candidates):
        wr = matchup_win_rate(cand, base_deck, model, device, quick_games, search_count)
        scored.append((wr, cand))
        print(f"  candidate {i+1}/{len(candidates)}: wr={wr:.3f}")

    scored.sort(key=lambda x: x[0], reverse=True)
    archive: list[list[int]] = [base_deck] + [deck for _, deck in scored[:k_archive - 1]]
    print(f"\nArchive: {len(archive)} decks (base + top {len(archive)-1} candidates)")

    print(f"\nBuilding {len(archive)}x{len(archive)} matchup matrix ({matrix_games} games/pair)...")
    M = build_matchup_matrix(archive, model, device, matrix_games, search_count)

    print("\nSolving Nash equilibrium...")
    weights = nash_mixture(M)
    best_idx = int(np.argmax(weights))
    best_deck = archive[best_idx]

    print(f"\nNash weights: {np.round(weights, 3)}")
    print(f"Best deck index: {best_idx} (weight={weights[best_idx]:.3f})")

    return best_deck, weights, archive
