# tests/test_league.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
from model.net import PolicyValueNet
from train.league import League, run_league

DECK = [677]*4+[678]*3+[1079]*4+[1086]*4+[1121]*4+[1082]*1+[1097]*2+[1123]*2+[1210]*4+[1190]*4+[1188]*2+[6]*26
DEVICE = torch.device('cpu')


def _make_league():
    main = PolicyValueNet()
    exploiter = PolicyValueNet()
    teacher = PolicyValueNet()
    return League(main, exploiter, teacher, DECK, DEVICE)


def test_league_play_game_returns_samples():
    league = _make_league()
    samples = league.play_game(matchup='main_vs_main', search_count=2, epsilon=1.0)
    assert len(samples) > 0, "play_game must return at least one sample"


def test_league_play_game_exploiter_matchup():
    league = _make_league()
    samples = league.play_game(matchup='main_vs_exploiter', search_count=2, epsilon=1.0)
    assert isinstance(samples, list)


def test_exploiter_update_copies_weights():
    league = _make_league()
    with torch.no_grad():
        for p in league.main.parameters():
            p.fill_(0.42)
    league.update_exploiter()
    for pm, pe in zip(league.main.parameters(), league.exploiter.parameters()):
        assert torch.allclose(pm, pe), "Exploiter must match main after update"


def test_teacher_weights_frozen():
    league = _make_league()
    original = {k: v.clone() for k, v in league.teacher.state_dict().items()}
    league.play_game(matchup='main_vs_teacher', search_count=2, epsilon=1.0)
    for k, v in league.teacher.state_dict().items():
        assert torch.allclose(v, original[k]), f"Teacher weight {k} changed!"


def test_ppo_train_step_returns_float():
    league = _make_league()
    optimizer = torch.optim.AdamW(league.main.parameters(), lr=3e-4)
    samples = league.play_game(matchup='main_vs_main', search_count=2, epsilon=1.0)
    if not samples:
        return
    from train.ppo import compute_upgo_returns
    terminal = 0.0
    upgo = compute_upgo_returns(samples, terminal_reward=terminal)
    loss = league.ppo_train_step(samples[:2], optimizer, upgo[:2])
    assert isinstance(loss, float)
    assert loss >= 0
