"""
Evaluation script for trained Halite III RL bots.

Runs N complete games of a model checkpoint against a scripted opponent and
reports win rate, mean final halite, mean halite-per-turn, and a per-game table.

Usage
-----
    # Evaluate a weights file against the default greedy opponent
    python rl_eval.py --model checkpoints_v9\model_final_weights.pt --games 20

    # Compare two checkpoints
    python rl_eval.py --model checkpoints_v9\model_ep500_weights.pt --games 20
    python rl_eval.py --model checkpoints_v9\model_ep1000_weights.pt --games 20

    # Use deterministic (greedy) actions instead of sampling
    python rl_eval.py --model checkpoints_v9\model_final_weights.pt --games 20 --deterministic

    # Change opponent
    python rl_eval.py --model checkpoints_v9\model_final_weights.pt --games 20 --opponent greedy
    python rl_eval.py --model checkpoints_v9\model_final_weights.pt --games 20 --opponent idle
"""

import argparse
import os
import sys
import random
from typing import Optional

import torch
import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MY_EXT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)     # rl_v1/ — finds rl_env, rl_model
sys.path.insert(0, _MY_EXT)  # my_extension/ — finds halite_engine

from rl_env   import HaliteEnv
from rl_model import ActorCritic


# ---------------------------------------------------------------------------
# Single game runner
# ---------------------------------------------------------------------------

def run_game(
    model:         ActorCritic,
    opponent:      str,
    width:         int,
    height:        int,
    seed:          Optional[int],
    deterministic: bool,
    device:        torch.device,
) -> dict:
    """
    Run one complete game and return result statistics.

    Returns
    -------
    dict with keys: rl_halite, opp_halite, winner, turns, halite_per_turn
    """
    env  = HaliteEnv(
        width          = width,
        height         = height,
        num_players    = 2,
        seed           = seed,
        opponent_policy= opponent,
    )
    obs, info = env.reset()
    done      = False

    while not done:
        spawn = (
            len(obs) < 8
            and env.engine.players[0]['energy'] >= 1000
            and env.engine.turn < env.engine.max_turns * 0.7
        )
        ship_actions = {}
        for ship_id, (spatial, scalars) in obs.items():
            sp_t = torch.from_numpy(spatial).to(device)
            sc_t = torch.from_numpy(scalars).to(device)
            if deterministic:
                action = model.greedy_action(sp_t, sc_t)
            else:
                action, _, _ = model.select_action(sp_t, sc_t)
            ship_actions[ship_id] = action

        obs, _, done, info = env.step(ship_actions, spawn=spawn)

    eng          = env.engine
    rl_halite    = eng.players[0]['energy']
    opp_halite   = eng.players[1]['energy'] if eng.num_players > 1 else 0
    turns        = eng.turn

    return {
        'rl_halite':      rl_halite,
        'opp_halite':     opp_halite,
        'winner':         'RL' if rl_halite > opp_halite else ('OPP' if opp_halite > rl_halite else 'TIE'),
        'turns':          turns,
        'halite_per_turn': rl_halite / max(turns, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(cfg: dict) -> None:
    device = torch.device(cfg['device'])

    # Load model
    model = ActorCritic().to(device)
    weights_path = cfg['model']
    if not os.path.isfile(weights_path):
        print(f"ERROR: model file not found: {weights_path}")
        sys.exit(1)

    ckpt = torch.load(weights_path, map_location=device)
    # Support both plain state_dict and full checkpoint dicts
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded model from {weights_path}")

    n_games   = cfg['games']
    opponent  = cfg['opponent']
    det       = cfg['deterministic']
    width     = cfg['width']
    height    = cfg['height']
    base_seed = cfg['seed']

    print(f"Running {n_games} games vs '{opponent}' opponent "
          f"({'deterministic' if det else 'stochastic'} actions) "
          f"on {width}×{height} maps\n")

    results   = []
    wins = ties = losses = 0

    print(f"{'Game':>5}  {'RL halite':>10}  {'Opp halite':>10}  {'Result':>6}  {'Turns':>5}  {'Hal/Turn':>8}")
    print("-" * 58)

    for i in range(1, n_games + 1):
        seed = (base_seed + i) if base_seed is not None else random.randint(0, 2**31)
        r    = run_game(model, opponent, width, height, seed, det, device)
        results.append(r)

        if r['winner'] == 'RL':
            wins   += 1
        elif r['winner'] == 'TIE':
            ties   += 1
        else:
            losses += 1

        print(f"{i:>5}  {r['rl_halite']:>10}  {r['opp_halite']:>10}  "
              f"{r['winner']:>6}  {r['turns']:>5}  {r['halite_per_turn']:>8.1f}")

    # Summary
    mean_rl  = sum(r['rl_halite']  for r in results) / n_games
    mean_opp = sum(r['opp_halite'] for r in results) / n_games
    mean_hpt = sum(r['halite_per_turn'] for r in results) / n_games
    win_rate = wins / n_games * 100

    print("\n" + "=" * 58)
    print(f"Results over {n_games} games:")
    print(f"  Win rate     : {win_rate:.1f}%  ({wins}W / {ties}T / {losses}L)")
    print(f"  Mean halite  : RL={mean_rl:.0f}  vs  Opp={mean_opp:.0f}")
    print(f"  Halite/turn  : {mean_hpt:.1f}")
    print(f"  RL advantage : {mean_rl - mean_opp:+.0f} halite on average")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained Halite III RL bot')
    parser.add_argument('--model',         required=True,         help='Path to *_weights.pt or full .pt checkpoint')
    parser.add_argument('--games',         type=int,   default=20, help='Number of games to run (default: 20)')
    parser.add_argument('--opponent',      default='greedy',       help='Opponent policy: greedy|idle|random (default: greedy)')
    parser.add_argument('--deterministic', action='store_true',    help='Use greedy actions instead of sampling')
    parser.add_argument('--width',         type=int,   default=32, help='Map width (default: 32)')
    parser.add_argument('--height',        type=int,   default=32, help='Map height (default: 32)')
    parser.add_argument('--device',        default='cpu',          help='cpu or cuda (default: cpu)')
    parser.add_argument('--seed',          type=int,   default=None, help='Base random seed (default: random per game)')

    args = parser.parse_args()
    cfg  = vars(args)
    evaluate(cfg)


if __name__ == '__main__':
    main()
