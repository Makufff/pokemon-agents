# tests/test_dmc.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.net import PolicyValueNet
from train.dmc import obs_to_tensors, eval_vs_random, self_play_game, train_step, mcts_step

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DEVICE = torch.device('cpu')

def test_self_play_game_returns_samples():
    model = PolicyValueNet()
    samples = self_play_game(DECK, model, DEVICE, epsilon=1.0, search_count=2)
    assert len(samples) > 0
    assert any(s.td_value != 0 for s in samples)

def test_train_step_returns_scalar_loss():
    model = PolicyValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    samples = self_play_game(DECK, model, DEVICE, epsilon=1.0, search_count=2)
    if not samples:
        return
    loss = train_step(samples[:1], model, optimizer, DEVICE)
    assert isinstance(loss, float)
    assert loss >= 0

def test_eval_vs_random_returns_winrate():
    model = PolicyValueNet()
    wr = eval_vs_random(DECK, model, DEVICE, n_games=2)
    assert 0.0 <= wr <= 1.0

def test_mcts_step_sets_log_prob_old():
    import math
    model = PolicyValueNet()
    deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
    device = torch.device('cpu')
    from core.env import PTCGEnv
    env = PTCGEnv()
    obs = env.reset(deck, deck, your_index=0)
    while obs.get('select') is None:
        obs, done, _ = env.step([])
    action, sample = mcts_step(obs, deck, model, device, search_count=2)
    env.close()
    if sample is not None:
        assert sample.log_prob_old <= 0.0, "log_prob must be non-positive (log of probability)"
        assert sample.log_prob_old > -50.0, "log_prob implausibly small"

def test_self_play_game_populates_opp_hand():
    model = PolicyValueNet()
    deck = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
    device = torch.device('cpu')
    samples = self_play_game(deck, model, device, epsilon=1.0, search_count=2)
    has_oracle = any(len(s.opp_hand_ids) > 0 for s in samples)
    assert has_oracle, "Some samples should have opponent hand IDs (oracle)"
