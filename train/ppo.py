# train/ppo.py
import torch
import torch.nn.functional as F

from train.buffer import LearnSample
from train.dmc import _pad, MAX_PAD_HAND, MAX_PAD_DISCARD, MAX_PAD_DECK


def compute_ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: float,
    advantage: float,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO surrogate loss for one (state, action) pair."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    adv = torch.tensor(advantage, dtype=torch.float32, device=new_log_prob.device)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.min(ratio * adv, clipped)


def compute_upgo_returns(
    samples: list[LearnSample],
    terminal_reward: float,
) -> list[float]:
    """UPGO targets: G_T = terminal_reward, G_t = max(V(s_{t+1}), G_{t+1})."""
    if not samples:
        return []
    n = len(samples)
    returns = [0.0] * n
    G = terminal_reward
    returns[-1] = G
    for i in range(n - 2, -1, -1):
        v_next = samples[i + 1].td_value
        G = max(v_next, G)
        returns[i] = G
    return returns


def compute_kl_loss(
    model,
    teacher,
    sample: LearnSample,
    device: torch.device,
) -> torch.Tensor:
    """KL(teacher || model) for one sample's action distribution."""
    board = torch.tensor(sample.board, device=device).unsqueeze(0)
    hand_t = _pad(sample.hand_ids, MAX_PAD_HAND).unsqueeze(0).to(device)
    discard_t = _pad(sample.discard_ids, MAX_PAD_DISCARD).unsqueeze(0).to(device)
    deck_t = _pad(sample.deck_ids, MAX_PAD_DECK).unsqueeze(0).to(device)
    scalars = torch.tensor(sample.scalars, device=device).unsqueeze(0)
    opt_types = torch.tensor(sample.opt_types, dtype=torch.long, device=device)
    opt_cards = torch.tensor(sample.opt_cards, dtype=torch.long, device=device)

    _, new_scores = model(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)
    with torch.no_grad():
        _, teacher_scores = teacher(board, hand_t, discard_t, deck_t, scalars, opt_types, opt_cards)

    new_log_probs = F.log_softmax(new_scores, dim=0)
    teacher_log_probs = F.log_softmax(teacher_scores, dim=0)
    teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - new_log_probs)).sum()
