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

def test_oracle_does_not_affect_scores():
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
    zero_oracle = torch.zeros(1, 5, dtype=torch.long)
    v_none, _ = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards, None)
    v_zero, _ = net(board, hand_ids, discard_ids, deck_ids, scalars, opt_types, opt_cards, zero_oracle)
    assert torch.allclose(v_none, v_zero, atol=1e-5), "None oracle must equal all-zero oracle"
