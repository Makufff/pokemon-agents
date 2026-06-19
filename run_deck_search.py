"""Phase 4: Run deck optimization. Outputs best deck to deck.csv.

Usage:
    python -u run_deck_search.py > logs/deck_search.log 2>&1
"""
import os
import csv
import torch

from model.net import PolicyValueNet
from deck.optimize import BASE_DECK, run_deck_search, validate_deck

SAVE_CSV = 'deck_optimized.csv'


def load_model(device: torch.device) -> PolicyValueNet:
    for path in ('model_phase2.pt', 'model.pt'):
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            net = PolicyValueNet().to(device)
            current_shapes = {k: v.shape for k, v in net.state_dict().items()}
            filtered = {k: v for k, v in ckpt['model'].items()
                        if k in current_shapes and v.shape == current_shapes[k]}
            net.load_state_dict(filtered, strict=False)
            net.eval()
            print(f'Loaded {path}')
            return net
        except Exception as e:
            print(f'Failed to load {path}: {e}')
    raise RuntimeError('No model checkpoint found')


def write_deck_csv(deck: list[int], path: str) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['card_id', 'count'])
        from collections import Counter
        for cid, cnt in sorted(Counter(deck).items()):
            writer.writerow([cid, cnt])
    print(f'Saved best deck to {path}')


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_model(device)

    ok, reason = validate_deck(BASE_DECK)
    assert ok, f'Base deck invalid: {reason}'

    best_deck, weights, archive = run_deck_search(
        base_deck=BASE_DECK,
        model=model,
        device=device,
        n_candidates=20,
        k_archive=8,
        quick_games=10,
        matrix_games=20,
        search_count=3,
        n_swaps=2,
        seed=42,
    )

    ok, reason = validate_deck(best_deck)
    print(f'Best deck valid: {ok} ({reason})')
    write_deck_csv(best_deck, SAVE_CSV)

    print('\nAll archive decks:')
    for i, (deck, w) in enumerate(zip(archive, weights)):
        print(f'  [{i}] weight={w:.3f} deck={sorted(set(deck))}')


if __name__ == '__main__':
    main()
