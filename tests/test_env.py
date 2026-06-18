import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.env import PTCGEnv

# Valid deck: 4x Riolu, 3x Mega Lucario ex, and trainers with energy
# Card 1082 is an ACE SPEC, so it can only have 1 copy (not 2)
SAMPLE_DECK = [677]*4 + [678]*3 + [1079]*4 + [1086]*4 + [1121]*4 + \
              [1082]*1 + [1097]*2 + [1123]*2 + [1210]*4 + \
              [1190]*4 + [1188]*2 + [6]*26

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

def test_close_is_idempotent():
    env = PTCGEnv()
    env.reset(SAMPLE_DECK, SAMPLE_DECK)
    env.close()
    env.close()  # Must not crash
