# tests/test_ppo.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import torch
from model.net import PolicyValueNet
from train.buffer import LearnSample
from train.ppo import compute_ppo_loss, compute_upgo_returns, compute_kl_loss


def _make_sample(v=0.5, n_opts=3):
    return LearnSample(
        board=np.zeros((12, 40), dtype=np.float32),
        hand_ids=[677], discard_ids=[], deck_ids=[6] * 20,
        scalars=np.zeros(8, dtype=np.float32),
        opt_types=list(range(n_opts)), opt_cards=[0] * n_opts,
        action_idx=0, td_value=v, mcts_policy=[1.0 / n_opts] * n_opts,
        log_prob_old=-math.log(n_opts), opp_hand_ids=[678],
    )


def test_ppo_loss_is_scalar():
    loss = compute_ppo_loss(
        new_log_prob=torch.tensor(-1.0),
        old_log_prob=-1.0,
        advantage=0.5,
    )
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() < 0  # positive advantage → negative loss (minimized → more negative)


def test_ppo_loss_clips_large_ratio():
    large_ratio_loss = compute_ppo_loss(
        new_log_prob=torch.tensor(-0.01),
        old_log_prob=-5.0,  # ratio = exp(4.99) >> 1
        advantage=1.0,
        clip_eps=0.2,
    )
    # Clipped: -min(ratio*1.0, 1.2*1.0) = -1.2
    assert abs(large_ratio_loss.item() - (-1.2)) < 0.01, f"Expected ~-1.2 clipped, got {large_ratio_loss.item()}"


def test_upgo_returns_propagates_terminal():
    samples = [_make_sample(v=0.2), _make_sample(v=0.6), _make_sample(v=0.9)]
    returns = compute_upgo_returns(samples, terminal_reward=1.0)
    assert len(returns) == 3
    assert returns[-1] == 1.0
    # t=1: max(V(s_2)=0.9, G_2=1.0) = 1.0
    assert returns[1] == 1.0
    # t=0: max(V(s_1)=0.6, G_1=1.0) = 1.0
    assert returns[0] == 1.0


def test_upgo_returns_uses_value_when_better():
    samples = [_make_sample(v=0.0), _make_sample(v=0.8)]
    returns = compute_upgo_returns(samples, terminal_reward=0.5)
    assert returns[-1] == 0.5
    # t=0: max(V(s_1)=0.8, G_1=0.5) = 0.8
    assert returns[0] == 0.8


def test_upgo_returns_empty():
    assert compute_upgo_returns([], terminal_reward=1.0) == []


def test_kl_loss_same_model_near_zero():
    model = PolicyValueNet()
    sample = _make_sample()
    device = torch.device('cpu')
    loss = compute_kl_loss(model, model, sample, device)
    assert loss.item() < 1e-4, f"KL(model, model) should be ~0, got {loss.item()}"


def test_kl_loss_different_models_positive():
    m1 = PolicyValueNet()
    m2 = PolicyValueNet()
    sample = _make_sample()
    device = torch.device('cpu')
    loss = compute_kl_loss(m1, m2, sample, device)
    assert loss.item() >= 0, "KL divergence must be non-negative"
