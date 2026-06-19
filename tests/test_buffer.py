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
        opt_types=[7, 14],
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

def test_phase2_fields_have_defaults():
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
