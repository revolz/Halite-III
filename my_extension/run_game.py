#!/usr/bin/env python3
"""
run_game.py  –  Run a Halite III game using the Python engine.

Usage examples:

  # Two copies of the starter-kit bot vs each other (32x32 map):
  python run_game.py

  # Custom map size and seed:
  python run_game.py --width 40 --height 40 --seed 12345

  # Supply your own bot commands:
  python run_game.py --bot "python MyBot.py" --bot "python MyBot.py"

  # 4-player game:
  python run_game.py --players 4

  # Verbose turn-by-turn output:
  python run_game.py --verbose

  # Save a replay file (auto-named by seed):
  python run_game.py --replay

  # Save a replay file to a specific path:
  python run_game.py --replay-file my_game.hlt
"""

import argparse
import datetime
import os
import sys

# Make sure halite_engine.py (in the same directory) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halite_engine import HaliteEngine

# Path to the Python3 starter kit (bots need access to the 'hlt' package)
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER_KIT  = os.path.join(REPO_ROOT, 'starter_kits', 'Python3')
DEFAULT_BOT  = f'python -u "{os.path.join(STARTER_KIT, "MyBot.py")}"'

# Directory where replay files are saved when using --replay (auto-naming)
REPLAYS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replays')


def main():
    parser = argparse.ArgumentParser(
        description='Run a Halite III game with the Python engine.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--width',   type=int, default=32,
                        help='Map width  (32/40/48/56/64, default 32)')
    parser.add_argument('--height',  type=int, default=32,
                        help='Map height (32/40/48/56/64, default 32)')
    parser.add_argument('--seed',    type=int, default=None,
                        help='Random seed for reproducible maps')
    parser.add_argument('--players', type=int, default=2,
                        help='Number of players (must match --bot count)')
    parser.add_argument('--bot',     action='append', dest='bots',
                        metavar='CMD',
                        help='Bot command (repeat for each player). '
                             'Defaults to the starter-kit MyBot.py.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print turn-by-turn progress')
    parser.add_argument('--replay',  action='store_true',
                        help='Save a replay .hlt file (auto-named, saved to '
                             'my_extension/replays/)')
    parser.add_argument('--replay-file', dest='replay_file', default=None,
                        metavar='PATH',
                        help='Save a replay .hlt file to the given path '
                             '(implies --replay)')
    args = parser.parse_args()

    # Build bot command list
    if args.bots:
        bots = args.bots
        # Adjust player count to match supplied bots if not explicitly set
        if args.players != len(bots) and len(bots) in (1, 2, 4):
            args.players = len(bots)
    else:
        # Default: run the starter-kit bot for each player
        bots = [DEFAULT_BOT] * args.players

    if len(bots) != args.players:
        parser.error(
            f"--players {args.players} but {len(bots)} --bot entries supplied"
        )

    # The starter-kit bot imports from './hlt', so we run it with the
    # starter_kits/Python3 directory on PYTHONPATH.
    import os as _os
    _os.environ['PYTHONPATH'] = (
        STARTER_KIT
        + _os.pathsep
        + _os.environ.get('PYTHONPATH', '')
    )

    print(f"=== Halite III Python Engine ===")
    print(f"Map: {args.width}x{args.height}  Players: {args.players}")
    if args.seed:
        print(f"Seed: {args.seed}")
    print()
    for i, bot_cmd in enumerate(bots):
        print(f"  Bot {i}: {bot_cmd}")
    print()

    engine = HaliteEngine(
        width=args.width,
        height=args.height,
        num_players=args.players,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Determine replay file path
    replay_path = args.replay_file
    if replay_path is None and args.replay:
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        os.makedirs(REPLAYS_DIR, exist_ok=True)
        replay_path = os.path.join(
            REPLAYS_DIR,
            f'replay-{ts}-{engine.seed}-{args.width}-{args.height}.hlt'
        )

    results = engine.run(bots, replay_file=replay_path)

    print()
    print(f"=== Final Results (turn {engine.turn}) ===")
    for rank, (pid, halite) in enumerate(results, 1):
        print(f"  #{rank}  Player {pid}: {halite:,} halite")

    winner_pid, winner_halite = results[0]
    print(f"\nWinner: Player {winner_pid} with {winner_halite:,} halite")

    if replay_path:
        print(f"\nReplay saved: {replay_path}")


if __name__ == '__main__':
    main()
