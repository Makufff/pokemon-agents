import numpy as np
from cg.api import all_card_data, to_observation_class, AreaType, OptionType

CARD_COUNT = 1268  # max cardId + 1
SLOT_FEATURES = 40
MAX_ACTIONS = 64   # max candidate actions enumerated per decision
_MAX_HP = 400.0

# Load card metadata once at import time
_all_cards = all_card_data()
CARD_TABLE = {c.cardId: c for c in _all_cards}


def _encode_slot(poke, is_active: bool, ps: dict) -> np.ndarray:
    """Encode a single board slot as a 40-float vector.

    is_active: True if this is the Active Spot (status conditions apply)
    ps: PlayerState dict (used to read status conditions for active slot)
    """
    feat = np.zeros(SLOT_FEATURES, dtype=np.float32)
    if poke is None:
        feat[0] = 1.0
        return feat

    hp = poke.get('hp', 0)
    max_hp = poke.get('maxHp', _MAX_HP) or _MAX_HP  # guard against 0
    feat[1] = hp / _MAX_HP
    feat[2] = (max_hp - hp) / _MAX_HP
    feat[3] = float(poke.get('appearThisTurn', False))

    for e in poke.get('energies', []):
        e_type = int(e)
        if 0 <= e_type <= 11:  # energy types 0-11 map to feat[4..15]
            feat[4 + e_type] += 0.1

    feat[16] = min(len(poke.get('energyCards', [])) / 5.0, 1.0)
    feat[17] = min(len(poke.get('tools', [])) / 3.0, 1.0)

    card = CARD_TABLE.get(poke['id'])
    if card:
        feat[18] = float(card.ex)
        feat[19] = float(card.tera)
        feat[20] = float(card.megaEx)
        feat[21] = float(card.basic)
        feat[22] = float(card.stage1)
        feat[23] = float(card.stage2)
        feat[24] = card.retreatCost / 5.0

    feat[25] = min(len(poke.get('preEvolution', [])) / 3.0, 1.0)

    # Status conditions — only active slot has them (from PlayerState)
    if is_active:
        feat[26] = float(ps.get('poisoned', False))
        feat[27] = float(ps.get('burned', False))
        feat[28] = float(ps.get('asleep', False))
        feat[29] = float(ps.get('paralyzed', False))
        feat[30] = float(ps.get('confused', False))

    return feat


def encode_board(obs: dict, your_index: int) -> np.ndarray:
    """Encode the full board as [12, 40] float32 tensor.

    Slot order: your_active, your_bench[0..4], opp_active, opp_bench[0..4]
    """
    state = obs['current']
    board = np.zeros((12, SLOT_FEATURES), dtype=np.float32)
    slot = 0
    for pi_offset in [0, 1]:
        pi = (your_index + pi_offset) % 2
        ps = state['players'][pi]
        # Active slot
        active_list = ps.get('active', [])
        active = active_list[0] if active_list else None
        board[slot] = _encode_slot(active, is_active=True, ps=ps)
        slot += 1
        # Bench slots (up to 5)
        bench = ps.get('bench', [])
        for j in range(5):
            poke = bench[j] if j < len(bench) else None
            board[slot] = _encode_slot(poke, is_active=False, ps=ps)
            slot += 1
    return board


def encode_sets(obs: dict, your_index: int, your_deck: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return card ID lists for hand, discard pile, and remaining deck."""
    ps = obs['current']['players'][your_index]
    hand = ps.get('hand') or []
    hand_ids = [c['id'] for c in hand]
    discard_ids = [c['id'] for c in ps.get('discard', [])]
    deck_count = ps.get('deckCount', 0)
    # Approximation: assumes draws come from end of deck; actual order unknown
    deck_ids = your_deck[:deck_count]
    return hand_ids, discard_ids, deck_ids


def encode_scalars(obs: dict, your_index: int) -> np.ndarray:
    """Encode 8 global scalar features as float32 array."""
    state = obs['current']
    opp_index = 1 - your_index
    your_ps = state['players'][your_index]
    opp_ps = state['players'][opp_index]

    scalars = np.array([
        min(state['turn'] / 30.0, 1.0),  # normalized turn, capped at 1.0
        float(state.get('firstPlayer', -1) == your_index),
        float(state.get('supporterPlayed', False)),
        float(state.get('energyAttached', False)),
        len(your_ps.get('prize', [])) / 3.0,
        len(opp_ps.get('prize', [])) / 3.0,
        your_ps.get('deckCount', 0) / 60.0,
        opp_ps.get('deckCount', 0) / 60.0,
    ], dtype=np.float32)
    return scalars


def encode_option(opt: dict, obs: dict, your_index: int) -> tuple[int, int]:
    """Return (option_type_id, card_id) for a single option dict.

    card_id is 0 when the option has no associated card (END, YES, NO, etc.).
    """
    otype = int(opt['type'])
    card_id = 0

    state = obs['current']
    ps = state['players'][your_index]

    try:
        match otype:
            case 7:  # PLAY — play card from hand
                hand = ps.get('hand') or []
                idx = opt.get('index', 0)
                if idx < len(hand):
                    card_id = hand[idx]['id']
            case 3 | 8 | 9 | 10 | 11:  # CARD, ATTACH, EVOLVE, ABILITY, DISCARD
                area = opt.get('area')
                idx = opt.get('index', 0)
                pi = opt.get('playerIndex', your_index)
                card_id = _card_id_from_area(obs, area, idx, pi)
            case 13:  # ATTACK — encode the attacking Pokemon's card ID
                active = ps.get('active', [])
                card_id = active[0]['id'] if active else 0
    except (IndexError, KeyError, TypeError):
        card_id = 0

    return otype, max(0, min(card_id, CARD_COUNT - 1))


def _card_id_from_area(obs: dict, area: int | None, index: int, player_index: int) -> int:
    if area is None:
        return 0
    state = obs['current']
    ps = state['players'][player_index]
    try:
        match area:
            case 2:  # HAND
                hand = ps.get('hand') or []
                return hand[index]['id'] if index < len(hand) else 0
            case 3:  # DISCARD
                discard = ps.get('discard', [])
                return discard[index]['id'] if index < len(discard) else 0
            case 4:  # ACTIVE
                active = ps.get('active', [])
                return active[index]['id'] if index < len(active) else 0
            case 5:  # BENCH
                bench = ps.get('bench', [])
                return bench[index]['id'] if index < len(bench) else 0
            case 6:  # PRIZE
                prize = ps.get('prize', [])
                p = prize[index] if index < len(prize) else None
                return p['id'] if p else 0
            case 7:  # STADIUM
                stadium = state.get('stadium', [])
                return stadium[index]['id'] if index < len(stadium) else 0
            case 1:  # DECK — cards visible during deck-search effects
                deck_list = (obs.get('select') or {}).get('deck') or []
                return deck_list[index]['id'] if index < len(deck_list) else 0
            case _:
                return 0
    except (IndexError, KeyError, TypeError):
        return 0


def enumerate_actions(obs: dict) -> list[list[int]]:
    """Return up to MAX_ACTIONS candidate action lists from the current observation.

    For maxCount=1, each action is [i] for each legal option index i.
    For maxCount>1, enumerate combinations up to MAX_ACTIONS.
    """
    sel = obs.get('select')
    if not sel:
        return [[]]

    n_opts = len(sel['option'])
    max_count = sel['maxCount']
    min_count = sel['minCount']

    if max_count == 1 or min_count == max_count == 1:
        return [[i] for i in range(n_opts)]

    # Multi-select: enumerate combinations greedily up to MAX_ACTIONS
    actions = []
    actual_count = min(max_count, n_opts)
    if actual_count < min_count:
        return [list(range(n_opts))]  # take all available options
    indices = list(range(actual_count))
    while len(actions) < MAX_ACTIONS:
        if all(i < n_opts for i in indices):
            actions.append(indices.copy())
        # Increment in combinatorial order
        for pos in range(actual_count - 1, -1, -1):
            if indices[pos] < n_opts - (actual_count - pos):
                indices[pos] += 1
                for j in range(pos + 1, actual_count):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break

    return actions if actions else [list(range(min(min_count, n_opts)))]
