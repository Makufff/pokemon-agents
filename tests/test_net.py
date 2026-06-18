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
