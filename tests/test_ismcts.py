import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from core.env import PTCGEnv
from core.belief import BeliefState
from model.net import PolicyValueNet
from search.ismcts import ismcts_step

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DEVICE = torch.device('cpu')

def _get_action_obs():
    env = PTCGEnv()
    obs = env.reset(DECK, DECK, your_index=0)
    for _ in range(200):
        if obs.get('select') is not None:
            return obs, env
        obs, done, _ = env.step([])
        if done:
            break
    return obs, env

def test_ismcts_step_returns_valid_action():
    model = PolicyValueNet(); model.eval()
    belief = BeliefState(DECK)
    obs, env = _get_action_obs()
    try:
        action, sample = ismcts_step(obs, DECK, model, DEVICE, belief, k=2, search_count=3)
        assert isinstance(action, list)
        assert all(isinstance(a, int) for a in action)
    finally:
        env.close()

def test_ismcts_step_respects_timeout():
    import time
    model = PolicyValueNet(); model.eval()
    belief = BeliefState(DECK)
    obs, env = _get_action_obs()
    try:
        t0 = time.time()
        action, _ = ismcts_step(obs, DECK, model, DEVICE, belief, k=10, search_count=50, timeout_secs=1.0)
        assert time.time() - t0 < 3.0
        assert isinstance(action, list)
    finally:
        env.close()

def test_ismcts_step_updates_belief():
    model = PolicyValueNet(); model.eval()
    belief = BeliefState(DECK)
    obs, env = _get_action_obs()
    pool_before = len(belief.pool_list())
    try:
        ismcts_step(obs, DECK, model, DEVICE, belief, k=1, search_count=2)
    finally:
        env.close()
    assert len(belief.pool_list()) <= pool_before
