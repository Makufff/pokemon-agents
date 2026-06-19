# main.py
import os
import random
import torch

from cg.api import to_observation_class

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
DECK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deck.csv')
KAGGLE_PATH = '/kaggle_simulations/agent/'
SEARCH_COUNT = 5
TIMEOUT_SECS = 3.0


def _read_deck() -> list[int]:
    path = DECK_PATH if os.path.exists(DECK_PATH) else KAGGLE_PATH + 'deck.csv'
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()][:60]


_DECK = _read_deck()


def _load_model():
    try:
        from model.net import PolicyValueNet
        net = PolicyValueNet()
        ckpt_path = MODEL_PATH if os.path.exists(MODEL_PATH) else KAGGLE_PATH + 'model.pt'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            net.load_state_dict(ckpt['model'])
        net.eval()
        return net
    except Exception:
        return None


_MODEL = _load_model()
_DEVICE = torch.device('cpu')


def agent(obs_dict: dict) -> list[int]:
    """Competition entry point — called once per decision."""
    obs = to_observation_class(obs_dict)

    if obs.select is None:
        return _DECK

    n_opts = len(obs.select.option)
    max_count = obs.select.maxCount
    fallback = random.sample(range(n_opts), min(max_count, n_opts))

    if _MODEL is None:
        return fallback

    try:
        import time
        from train.dmc import mcts_step
        t0 = time.time()
        action, _ = mcts_step(obs_dict, _DECK, _MODEL, _DEVICE, search_count=SEARCH_COUNT)
        if time.time() - t0 > TIMEOUT_SECS:
            return fallback
        return action
    except Exception:
        return fallback
