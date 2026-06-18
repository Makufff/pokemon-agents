import numpy as np
from cg.api import all_card_data, to_observation_class, AreaType, OptionType

CARD_COUNT = 1268  # max cardId + 1
SLOT_FEATURES = 40
MAX_ACTIONS = 64   # max candidate actions enumerated per decision

# Load card metadata once at import time
_all_cards = all_card_data()
CARD_TABLE = {c.cardId: c for c in _all_cards}


def _encode_slot(poke, is_active: bool, ps_special: dict) -> np.ndarray:
    """Encode a single board slot (Pokemon or empty) as a 40-float vector."""
    feat = np.zeros(SLOT_FEATURES, dtype=np.float32)
    if poke is None:
        feat[0] = 1.0  # is_empty
        return feat

    feat[1] = poke['hp'] / 400.0
    max_hp = poke['maxHp']
    feat[2] = (max_hp - poke['hp']) / 400.0
    feat[3] = float(poke.get('appearThisTurn', False))

    for e in poke.get('energies', []):
        idx = 4 + int(e)
        if idx < 16:
            feat[idx] += 0.1

    feat[16] = len(poke.get('energyCards', [])) / 5.0
    feat[17] = len(poke.get('tools', [])) / 3.0

    card = CARD_TABLE.get(poke['id'])
    if card:
        feat[18] = float(card.ex)
        feat[19] = float(card.tera)
        feat[20] = float(card.megaEx)
        feat[21] = float(card.basic)
        feat[22] = float(card.stage1)
        feat[23] = float(card.stage2)
        feat[24] = card.retreatCost / 5.0

    feat[25] = len(poke.get('preEvolution', [])) / 3.0
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
        board[slot] = _encode_slot(active, is_active=True, ps_special=ps)
        slot += 1
        # Bench slots (up to 5)
        bench = ps.get('bench', [])
        for j in range(5):
            poke = bench[j] if j < len(bench) else None
            board[slot] = _encode_slot(poke, is_active=False, ps_special=ps)
            slot += 1
    return board


def encode_sets(obs: dict, your_index: int, your_deck: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return card ID lists for hand, discard pile, and remaining deck."""
    ps = obs['current']['players'][your_index]
    hand = ps.get('hand') or []
    hand_ids = [c['id'] for c in hand]
    discard_ids = [c['id'] for c in ps.get('discard', [])]
    deck_count = ps.get('deckCount', 0)
    deck_ids = your_deck[:deck_count]
    return hand_ids, discard_ids, deck_ids


def encode_scalars(obs: dict, your_index: int) -> np.ndarray:
    """Encode 8 global scalar features as float32 array."""
    state = obs['current']
    opp_index = 1 - your_index
    your_ps = state['players'][your_index]
    opp_ps = state['players'][opp_index]

    scalars = np.array([
        state['turn'] / 10.0,
        float(state.get('firstPlayer', -1) == your_index),
        float(state.get('supporterPlayed', False)),
        float(state.get('energyAttached', False)),
        len(your_ps.get('prize', [])) / 6.0,
        len(opp_ps.get('prize', [])) / 6.0,
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
            case 13:  # ATTACK — use attackId as proxy (clamped)
                card_id = min(opt.get('attackId', 0), CARD_COUNT - 1)
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
    indices = list(range(max_count))
    while len(actions) < MAX_ACTIONS:
        if all(i < n_opts for i in indices):
            actions.append(indices.copy())
        # Increment in combinatorial order
        for pos in range(max_count - 1, -1, -1):
            if indices[pos] < n_opts - (max_count - pos):
                indices[pos] += 1
                for j in range(pos + 1, max_count):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break

    return actions if actions else [[0]]
