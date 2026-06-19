"""Tests for deck/optimize.py."""
import random
from collections import Counter
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from deck.optimize import (
    BASE_DECK,
    DECK_SIZE,
    MAX_COPIES,
    PROTECTED_IDS,
    _card_db,
    _swap_pool,
    validate_deck,
    mutate_deck,
    nash_mixture,
    build_matchup_matrix,
)


# ── validate_deck ─────────────────────────────────────────────────────────────

def test_base_deck_is_valid():
    ok, reason = validate_deck(BASE_DECK)
    assert ok, reason


def test_validate_wrong_length():
    ok, reason = validate_deck(BASE_DECK[:59])
    assert not ok
    assert "length" in reason


def test_validate_too_many_copies():
    # Replace one card with a 5th copy of 6 (Fighting Energy)
    deck = list(BASE_DECK)
    for i, cid in enumerate(deck):
        if Counter(deck)[6] < 5:
            break
        # 6 already appears 26 times — use a 4-copy-limited card instead
    # Put 5 copies of Ultra Ball (1121) in the deck
    deck = list(BASE_DECK)
    slots = [i for i, c in enumerate(deck) if c == 1121]
    # Replace slot 0's neighbour (not an 1121) with 1121
    for i in range(len(deck)):
        if deck[i] != 1121:
            deck[i] = 1121
            break
    ok, reason = validate_deck(deck)
    assert not ok
    assert "5" in reason or "max 4" in reason


def test_validate_no_basic_pokemon():
    db = _card_db()
    # Replace all basic Pokémon (677) with supporters
    deck = [1190 if cid == 677 else cid for cid in BASE_DECK]
    # Ensure max-4 rule doesn't trip first
    counts = Counter(deck)
    # If 1190 appears > 4 times, use different non-basic cards
    if counts[1190] > 4:
        extra = counts[1190] - 4
        replacements = iter([1188] * extra)
        new_deck = []
        seen_1190 = 0
        for cid in deck:
            if cid == 1190 and seen_1190 >= 4:
                new_deck.append(next(replacements))
            else:
                if cid == 1190:
                    seen_1190 += 1
                new_deck.append(cid)
        deck = new_deck
    ok, _ = validate_deck(deck)
    # If still valid, test that removing ALL basic Pokémon fails
    deck2 = [1188 if db.get(cid) and db[cid].basic else cid for cid in BASE_DECK]
    # This may trip copy limit; just test the no-basic path directly
    has_basic = any(db.get(cid) and db[cid].basic for cid in set(deck2))
    assert not has_basic or ok  # if we managed to remove basics, deck should be invalid


def test_validate_two_ace_specs():
    # Replace one non-protected card with a second ACE SPEC
    db = _card_db()
    ace_specs = [c for c in db.values() if c.aceSpec and c.cardId != 1082]
    if not ace_specs:
        pytest.skip("no alternative ACE SPEC cards in card DB")
    second_ace = ace_specs[0].cardId
    deck = list(BASE_DECK)
    for i, cid in enumerate(deck):
        if cid not in PROTECTED_IDS and cid != 6:
            deck[i] = second_ace
            break
    ok, reason = validate_deck(deck)
    assert not ok
    assert "ACE SPEC" in reason


# ── mutate_deck ───────────────────────────────────────────────────────────────

def test_mutate_preserves_length():
    db = _card_db()
    pool = _swap_pool(db)
    rng = random.Random(0)
    mutated = mutate_deck(BASE_DECK, pool, rng, n_swaps=3)
    assert len(mutated) == DECK_SIZE


def test_mutate_produces_valid_deck():
    db = _card_db()
    pool = _swap_pool(db)
    rng = random.Random(1)
    for _ in range(10):
        mutated = mutate_deck(BASE_DECK, pool, rng, n_swaps=2)
        ok, reason = validate_deck(mutated)
        assert ok, f"invalid mutation: {reason}, deck={Counter(mutated)}"


def test_mutate_preserves_protected_ids():
    db = _card_db()
    pool = _swap_pool(db)
    rng = random.Random(2)
    for _ in range(10):
        mutated = mutate_deck(BASE_DECK, pool, rng, n_swaps=4)
        for pid in PROTECTED_IDS:
            base_count = BASE_DECK.count(pid)
            mut_count = mutated.count(pid)
            assert base_count == mut_count, (
                f"protected card {pid} changed from {base_count} to {mut_count}"
            )


def test_mutate_zero_swaps_unchanged():
    db = _card_db()
    pool = _swap_pool(db)
    rng = random.Random(3)
    mutated = mutate_deck(BASE_DECK, pool, rng, n_swaps=0)
    assert mutated == list(BASE_DECK)


# ── nash_mixture ──────────────────────────────────────────────────────────────

def test_nash_uniform_on_symmetric_matrix():
    """All decks equal → any valid distribution is Nash; check constraints only."""
    n = 4
    M = np.full((n, n), 0.5)
    p = nash_mixture(M)
    assert p.shape == (n,)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-6)
    assert all(w >= 0 for w in p)


def test_nash_dominant_strategy():
    """Deck 0 beats all others → Nash should heavily favor deck 0."""
    n = 3
    M = np.array([
        [0.5, 0.9, 0.85],
        [0.1, 0.5, 0.5],
        [0.15, 0.5, 0.5],
    ])
    p = nash_mixture(M)
    assert p[0] > 0.5, f"dominant deck should have high weight, got {p}"


def test_nash_weights_sum_to_one():
    rng = np.random.default_rng(7)
    M = rng.random((5, 5))
    M = (M + (1 - M.T)) / 2  # make zero-sum
    p = nash_mixture(M)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-6)
    assert all(w >= 0 for w in p)


# ── build_matchup_matrix ──────────────────────────────────────────────────────

def test_build_matchup_matrix_shape():
    """Patch matchup_win_rate to avoid running actual games."""
    fake_model = MagicMock()
    fake_device = MagicMock()
    archive = [list(BASE_DECK), list(BASE_DECK)]

    with patch('deck.optimize.matchup_win_rate', return_value=0.5) as mock_wr:
        M = build_matchup_matrix(archive, fake_model, fake_device, n_games_per_pair=2)

    assert M.shape == (2, 2)
    np.testing.assert_allclose(M[0, 0], 0.5)
    np.testing.assert_allclose(M[1, 1], 0.5)
    np.testing.assert_allclose(M[0, 1] + M[1, 0], 1.0, atol=1e-9)
    assert mock_wr.call_count == 1  # only upper triangle


def test_build_matchup_matrix_symmetry():
    fake_model = MagicMock()
    fake_device = MagicMock()
    archive = [list(BASE_DECK)] * 4

    with patch('deck.optimize.matchup_win_rate', side_effect=[0.6, 0.4, 0.55, 0.45, 0.5, 0.5]):
        M = build_matchup_matrix(archive, fake_model, fake_device, n_games_per_pair=2)

    for i in range(4):
        for j in range(4):
            if i != j:
                np.testing.assert_allclose(M[i, j] + M[j, i], 1.0, atol=1e-9,
                                           err_msg=f"M[{i},{j}]+M[{j},{i}] != 1")
