# main.py
import os
import random
import torch

from cg.api import to_observation_class

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_phase2.pt')
MODEL_PATH_FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
DECK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deck.csv')
KAGGLE_PATH = '/kaggle_simulations/agent/'
K_DETERMINIZATIONS = 3
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
        for ckpt_candidate in [MODEL_PATH, MODEL_PATH_FALLBACK,
                                KAGGLE_PATH + 'model_phase2.pt',
                                KAGGLE_PATH + 'model.pt']:
            if os.path.exists(ckpt_candidate):
                ckpt = torch.load(ckpt_candidate, map_location='cpu', weights_only=False)
                current_shapes = {k: v.shape for k, v in net.state_dict().items()}
                filtered = {k: v for k, v in ckpt['model'].items()
                            if k in current_shapes and v.shape == current_shapes[k]}
                net.load_state_dict(filtered, strict=False)
                break
        net.eval()
        return net
    except Exception:
        return None


_MODEL = _load_model()
_DEVICE = torch.device('cpu')
_BELIEF = None  # BeliefState reset at game start, persists within game


def agent(obs_dict: dict) -> list[int]:
    """Competition entry point -- called once per decision."""
    global _BELIEF

    obs = to_observation_class(obs_dict)

    if obs.select is None:
        from core.belief import BeliefState
        _BELIEF = BeliefState(_DECK)
        return _DECK

    n_opts = len(obs.select.option)
    max_count = obs.select.maxCount
    fallback = random.sample(range(n_opts), min(max_count, n_opts))

    if _MODEL is None:
        return fallback

    if _BELIEF is None:
        from core.belief import BeliefState
        _BELIEF = BeliefState(_DECK)

    try:
        import time
        from search.ismcts import ismcts_step
        t0 = time.time()
        action, _ = ismcts_step(
            obs_dict, _DECK, _MODEL, _DEVICE, _BELIEF,
            k=K_DETERMINIZATIONS,
            search_count=SEARCH_COUNT,
            timeout_secs=TIMEOUT_SECS,
        )
        if time.time() - t0 > TIMEOUT_SECS:
            return fallback
        return action if action else fallback
    except Exception:
        return fallback
