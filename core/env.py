from cg.game import battle_start, battle_finish, battle_select


class PTCGEnv:
    """Gym-like wrapper around libcg.so battle API."""

    _ERROR_MSGS = {
        1: "Invalid card ID in deck",
        2: "More than 4 copies of a named card",
        3: "No Basic Pokémon in deck",
        4: "More than 1 ACE SPEC card",
    }

    def __init__(self):
        self._your_index = 0

    def reset(self, deck0: list[int], deck1: list[int], your_index: int = 0) -> dict:
        """Start a new game. Returns the first observation dict."""
        self._your_index = your_index
        obs, start_data = battle_start(deck0, deck1)
        if start_data.errorPlayer >= 0:
            msg = self._ERROR_MSGS.get(start_data.errorType, "unknown deck error")
            raise ValueError(f"Player {start_data.errorPlayer} deck error: {msg}")
        return obs

    def step(self, action: list[int]) -> tuple[dict, bool, dict]:
        """Apply action indices. Returns (next_obs, done, info).

        Reward is NOT computed here — the training loop handles TD(λ) returns.
        info['result'] is set to the winner index (0/1) or 2 for draw when done.
        """
        obs = battle_select(action)
        result = obs['current']['result']
        done = result >= 0
        return obs, done, {'result': result, 'your_index': self._your_index}

    def close(self):
        """Free game memory."""
        battle_finish()
