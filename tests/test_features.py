import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.env import PTCGEnv
from core.features import encode_board, encode_sets, encode_scalars, enumerate_actions, encode_option

SAMPLE_DECK = [677]*4 + [678]*3 + [1079]*4 + [1086]*4 + [1121]*4 + \
              [1082]*1 + [1097]*2 + [1123]*2 + [1210]*4 + \
              [1190]*4 + [1188]*2 + [6]*26

def _get_first_real_obs():
    """Get first obs where select is not None and current is not None."""
    import random
    env = PTCGEnv()
    obs = env.reset(SAMPLE_DECK, SAMPLE_DECK, your_index=0)
    # Step through until we have a real game state (select may be None initially)
    for _ in range(200):
        if obs.get('select') is not None and obs.get('current') is not None:
            break
        n_opts = len(obs['select']['option']) if obs.get('select') else 1
        max_count = obs['select']['maxCount'] if obs.get('select') else 1
        import random
        action = random.sample(range(n_opts), min(max_count, n_opts))
        obs, done, _ = env.step(action)
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
