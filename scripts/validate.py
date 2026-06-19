"""Phase 5: Pre-submission validation — smoke test via agent() entry point.

Simulates the Kaggle evaluation loop:
  1. Calls agent() with select=None to get the deck.
  2. Plays N full games using PTCGEnv; calls agent() for every decision.
  3. Reports win/loss/draw stats and any exceptions.

Usage:
    python scripts/validate.py [--games N] [--timeout T]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _reload_agent():
    """Import (or re-import) main.agent fresh so globals reset between test runs."""
    if 'main' in sys.modules:
        # Reset module-level globals that persist across games
        mod = sys.modules['main']
        for attr in ('_MODEL', '_DECK', '_BELIEF'):
            if hasattr(mod, attr):
                setattr(mod, attr, None)
        importlib.reload(mod)
    else:
        importlib.import_module('main')
    return sys.modules['main'].agent


def run_one_game(agent_fn, timeout_secs: float) -> dict:
    """Run one game through the agent() entry point. Returns result dict."""
    from core.env import PTCGEnv

    # Step 1: deck selection — Kaggle passes obs with select=None, logs=[], current=None
    t0 = time.time()
    deck = agent_fn({'select': None, 'logs': [], 'current': None})
    assert isinstance(deck, list) and len(deck) == 60, (
        f'Initial agent() must return 60-card list, got {type(deck)} len={len(deck)}'
    )

    env = PTCGEnv()
    obs = env.reset(deck, deck, your_index=0)
    done = False
    n_decisions = 0
    total_agent_ms = 0.0

    try:
        while not done:
            sel = obs.get('select')
            if sel is None:
                obs, done, info = env.step([])
                continue

            t_agent = time.time()
            action = agent_fn(obs)
            agent_ms = (time.time() - t_agent) * 1000
            total_agent_ms += agent_ms

            if agent_ms / 1000 > timeout_secs:
                print(f'  WARNING: agent took {agent_ms:.0f} ms (limit {timeout_secs*1000:.0f} ms)')

            n_opts = len(sel['option'])
            max_c = sel['maxCount']
            min_c = sel['minCount']
            assert isinstance(action, list), f'action must be list, got {type(action)}'
            assert min_c <= len(action) <= max_c, (
                f'action length {len(action)} outside [{min_c}, {max_c}]'
            )
            assert all(0 <= a < n_opts for a in action), (
                f'action index out of range: {action}, n_opts={n_opts}'
            )
            assert len(set(action)) == len(action), f'duplicate action indices: {action}'

            obs, done, info = env.step(action)
            n_decisions += 1

        elapsed = time.time() - t0
        return {
            'result': info['result'],
            'decisions': n_decisions,
            'elapsed_s': elapsed,
            'avg_agent_ms': total_agent_ms / max(n_decisions, 1),
            'error': None,
        }
    except Exception as e:
        return {
            'result': -1,
            'decisions': n_decisions,
            'elapsed_s': time.time() - t0,
            'avg_agent_ms': 0.0,
            'error': traceback.format_exc(),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=5, help='Number of games to play')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='Per-decision timeout warning threshold (seconds)')
    args = parser.parse_args()

    print(f'=== Pre-submission Validation ({args.games} games) ===\n')

    agent_fn = _reload_agent()

    results = {0: 0, 1: 0, 2: 0, -1: 0}  # 0=p0 wins, 1=p1 wins, 2=draw, -1=error
    total_decisions = 0
    total_elapsed = 0.0
    total_agent_ms = 0.0

    for g in range(1, args.games + 1):
        r = run_one_game(agent_fn, args.timeout)
        results[r['result']] += 1
        total_decisions += r['decisions']
        total_elapsed += r['elapsed_s']
        total_agent_ms += r['avg_agent_ms']

        status = {0: 'P0 wins', 1: 'P1 wins', 2: 'draw', -1: 'ERROR'}[r['result']]
        print(f'Game {g:2d}: {status:8s} | {r["decisions"]:3d} decisions '
              f'| {r["elapsed_s"]:.1f}s total | {r["avg_agent_ms"]:.0f} ms/decision')
        if r['error']:
            print(f'  Error:\n{r["error"]}')

    n = args.games
    print(f'\n=== Summary ===')
    print(f'P0 wins: {results[0]}/{n}  P1 wins: {results[1]}/{n}  '
          f'Draws: {results[2]}/{n}  Errors: {results[-1]}/{n}')
    print(f'Avg decisions/game: {total_decisions/n:.0f}')
    print(f'Avg game time: {total_elapsed/n:.1f}s')
    print(f'Avg agent ms/decision: {total_agent_ms/n:.0f}')

    if results[-1] > 0:
        print('\nFAIL: errors detected — fix before submitting')
        sys.exit(1)
    else:
        print('\nOK: all games completed without errors')


if __name__ == '__main__':
    main()
