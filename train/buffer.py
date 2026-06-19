import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LearnSample:
    board: np.ndarray          # [12, 40]
    hand_ids: list[int]
    discard_ids: list[int]
    deck_ids: list[int]
    scalars: np.ndarray        # [8]
    opt_types: list[int]       # one per candidate action
    opt_cards: list[int]
    action_idx: int            # index of selected action
    td_value: float            # TD(λ) return
    mcts_policy: list[float]   # MCTS visit proportions per candidate action
    log_prob_old: float = 0.0
    opp_hand_ids: list[int] = field(default_factory=list)


class RingBuffer:
    """Fixed-capacity FIFO ring buffer for LearnSample objects."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque[LearnSample] = deque(maxlen=capacity)

    def push(self, sample: LearnSample) -> None:
        self._buf.append(sample)

    def sample(self, n: int) -> list[LearnSample]:
        if n > len(self._buf):
            raise ValueError(f"Cannot sample {n} from buffer of size {len(self._buf)}")
        return random.sample(list(self._buf), n)

    def __len__(self) -> int:
        return len(self._buf)
