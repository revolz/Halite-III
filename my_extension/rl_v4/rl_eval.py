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
from rl_features import target_dropoffs, spawn_econ_ok


# ---------------------------------------------------------------------------
# Single game runner — vs a scripted opponent (greedy / idle / random)
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
    Run one complete game vs a scripted opponent and return result statistics.

    Returns
    -------
    dict with keys: rl_halite, opp_halite, winner, turns, halite_per_turn,
                    dropoffs, opp_dropoffs
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
        eng = env.engine
        tgt = target_dropoffs(eng.width, eng.height)
        spawn = spawn_econ_ok(eng.players[0]['energy'], len(obs),
                              eng.max_turns - eng.turn,
                              len(eng.players[0]['dropoffs']), tgt)
        ship_actions = {}
        for ship_id, (spatial, scalars, mask) in obs.items():
            sp_t = torch.from_numpy(spatial).to(device)
            sc_t = torch.from_numpy(scalars).to(device)
            if deterministic:
                action = model.greedy_action(sp_t, sc_t, mask=mask)
            else:
                action, _, _ = model.select_action(sp_t, sc_t, mask=mask)
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
        'dropoffs':       len(eng.players[0]['dropoffs']),
        'opp_dropoffs':   len(eng.players[1]['dropoffs']) if eng.num_players > 1 else 0,
    }


# ---------------------------------------------------------------------------
# Head-to-head runner — rl_v4 vs another RL bot (rl_v3 / rl_v2) via the hlt
# protocol, so each bot observes through its OWN feature/action module.
# ---------------------------------------------------------------------------

_BOT_DIRS = {
    'rl_v3': os.path.join(_MY_EXT, 'rl_v3'),
    'rl_v2': os.path.join(_MY_EXT, 'rl_v2'),
}


def _bot_cmd(bot_dir: str, deterministic: bool) -> str:
    bot_py = os.path.join(bot_dir, 'rl_bot.py')
    weights = os.path.join(bot_dir, 'checkpoints', 'model_final_weights.pt')
    det = ' --deterministic' if deterministic else ''
    return f'python -u "{bot_py}" --model "{weights}"{det}'


def run_headtohead(
    model_path:    str,
    opponent_bot:  str,
    width:         int,
    height:        int,
    seed:          Optional[int],
    deterministic: bool,
) -> dict:
    """Run rl_v4 (player 0) vs `opponent_bot` (player 1) over the shared engine."""
    from halite_engine import HaliteEngine

    v4_dir   = _HERE
    v4_py     = os.path.join(v4_dir, 'rl_bot.py')
    det       = ' --deterministic' if deterministic else ''
    bot0_cmd  = f'python -u "{v4_py}" --model "{model_path}"{det}'
    bot1_cmd  = _bot_cmd(_BOT_DIRS[opponent_bot], deterministic)

    eng = HaliteEngine(width=width, height=height, num_players=2,
                       seed=seed, verbose=False)
    results = eng.run([bot0_cmd, bot1_cmd], replay_file=None)
    banks = {pid: e for pid, e in results}
    rl_halite, opp_halite = banks.get(0, 0), banks.get(1, 0)

    return {
        'rl_halite':      rl_halite,
        'opp_halite':     opp_halite,
        'winner':         'RL' if rl_halite > opp_halite else ('OPP' if opp_halite > rl_halite else 'TIE'),
        'turns':          eng.turn,
        'halite_per_turn': rl_halite / max(eng.turn, 1),
        'dropoffs':       eng._number_dropoffs.get(0, 0),
        'opp_dropoffs':   eng._number_dropoffs.get(1, 0),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(cfg: dict) -> None:
    device = torch.device(cfg['device'])

    n_games   = cfg['games']
    opponent  = cfg['opponent']
    det       = cfg['deterministic']
    width     = cfg['width']
    height    = cfg['height']
    base_seed = cfg['seed']
    weights_path = cfg['model']

    head_to_head = opponent in _BOT_DIRS

    # In scripted mode the model runs in-process; in head-to-head mode each bot
    # is a subprocess that loads its own weights, so only validate the path here.
    model = None
    if not os.path.isfile(weights_path):
        print(f"ERROR: model file not found: {weights_path}")
        sys.exit(1)
    if not head_to_head:
        model = ActorCritic().to(device)
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print(f"Loaded model from {weights_path}")
    else:
        print(f"Head-to-head: rl_v4 ({weights_path}) vs {opponent} (its own weights)")

    print(f"Running {n_games} games vs '{opponent}' "
          f"({'deterministic' if det else 'stochastic'} actions) "
          f"on {width}×{height} maps\n")

    results   = []
    wins = ties = losses = 0

    print(f"{'Game':>5}  {'RL halite':>10}  {'Opp halite':>10}  {'Result':>6}  "
          f"{'Turns':>5}  {'Drops':>5}  {'Hal/Turn':>8}")
    print("-" * 70)

    for i in range(1, n_games + 1):
        seed = (base_seed + i) if base_seed is not None else random.randint(0, 2**31)
        if head_to_head:
            r = run_headtohead(weights_path, opponent, width, height, seed, det)
        else:
            r = run_game(model, opponent, width, height, seed, det, device)
        results.append(r)

        if r['winner'] == 'RL':
            wins   += 1
        elif r['winner'] == 'TIE':
            ties   += 1
        else:
            losses += 1

        print(f"{i:>5}  {r['rl_halite']:>10}  {r['opp_halite']:>10}  "
              f"{r['winner']:>6}  {r['turns']:>5}  {r['dropoffs']:>5}  "
              f"{r['halite_per_turn']:>8.1f}")

    # Summary
    mean_rl   = sum(r['rl_halite']  for r in results) / n_games
    mean_opp  = sum(r['opp_halite'] for r in results) / n_games
    mean_hpt  = sum(r['halite_per_turn'] for r in results) / n_games
    mean_drop = sum(r['dropoffs'] for r in results) / n_games
    win_rate  = wins / n_games * 100

    print("\n" + "=" * 70)
    print(f"Results over {n_games} games:")
    print(f"  Win rate      : {win_rate:.1f}%  ({wins}W / {ties}T / {losses}L)")
    print(f"  Mean halite   : RL={mean_rl:.0f}  vs  Opp={mean_opp:.0f}")
    print(f"  Halite/turn   : {mean_hpt:.1f}")
    print(f"  Dropoffs/game : {mean_drop:.2f}")
    print(f"  RL advantage  : {mean_rl - mean_opp:+.0f} halite on average")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained Halite III RL bot')
    parser.add_argument('--model',         required=True,         help='Path to *_weights.pt or full .pt checkpoint')
    parser.add_argument('--games',         type=int,   default=20, help='Number of games to run (default: 20)')
    parser.add_argument('--opponent',      default='greedy',       help='Opponent: greedy|idle|random (scripted) or rl_v3|rl_v2 (head-to-head). Default: greedy')
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
