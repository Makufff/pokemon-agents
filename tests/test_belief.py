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
