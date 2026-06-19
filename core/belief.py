# core/belief.py
import random
from collections import Counter

_BASIC_POKEMON_ID = 677  # Riolu fallback


class BeliefState:
    """Card-counting belief model for the opponent's hidden cards.

    Prior: assumes opponent deck = our own deck (mirror match assumption).
    After observing a card in a public zone, removes it from the unknown pool once.
    """

    def __init__(self, our_deck: list[int]):
        self._prior = list(our_deck)
        self._prior_counter = Counter(our_deck)
        self._pool = Counter(our_deck)
        self._public_seen: Counter = Counter()

    def reset(self, our_deck: list[int]) -> None:
        self._prior = list(our_deck)
        self._prior_counter = Counter(our_deck)
        self._pool = Counter(our_deck)
        self._public_seen = Counter()

    def pool_list(self) -> list[int]:
        return list(self._pool.elements())

    def mark_public(self, card_ids: list[int]) -> None:
        """Mark card IDs as observed in a public zone (idempotent snapshot merge).

        Treats card_ids as a cumulative snapshot: updates _public_seen using max
        per-card count, then rebuilds the pool. Repeated calls with the same list
        have no further effect.
        """
        self._rebuild_pool(card_ids)

    def update_from_obs(self, obs_class, opp_idx: int) -> None:
        """Extract public opponent cards from observation and update pool."""
        if obs_class.current is None:
            return
        opp_ps = obs_class.current.players[opp_idx]
        public: list[int] = []
        for card in opp_ps.discard:
            public.append(card.id)
        for poke in opp_ps.active:
            if poke is not None:
                public.append(poke.id)
                for c in poke.energyCards + poke.tools + poke.preEvolution:
                    public.append(c.id)
        for poke in opp_ps.bench:
            public.append(poke.id)
            for c in poke.energyCards + poke.tools + poke.preEvolution:
                public.append(c.id)
        for p in opp_ps.prize:
            if p is not None:
                public.append(p.id)
        self._rebuild_pool(public)

    def _rebuild_pool(self, public_cards: list[int]) -> None:
        new_seen = Counter(public_cards)
        for cid, count in new_seen.items():
            self._public_seen[cid] = max(self._public_seen[cid], count)
        self._pool = Counter(self._prior)
        for cid, seen in self._public_seen.items():
            remove = min(seen, self._prior_counter[cid])
            self._pool[cid] -= remove
            if self._pool[cid] <= 0:
                del self._pool[cid]

    def sample_determinization(
        self, hand_count: int, deck_count: int, prize_count: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Sample one possible distribution of opponent's hidden cards.

        Pads with basic energy (ID 6) if pool is smaller than needed.
        Guarantees deck has at least one Basic Pokemon (677 = Riolu).
        """
        available = list(self._pool.elements())
        random.shuffle(available)
        n_needed = hand_count + deck_count + prize_count
        while len(available) < n_needed:
            available.append(6)
        hand = available[:hand_count]
        deck = available[hand_count:hand_count + deck_count]
        prize = available[hand_count + deck_count:hand_count + deck_count + prize_count]
        if deck_count > 0 and _BASIC_POKEMON_ID not in deck:
            deck[0] = _BASIC_POKEMON_ID
        return hand, deck, prize
