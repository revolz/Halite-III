#!/usr/bin/env python3
"""
rl_v7 / rl_bot.py  --  live inference bot.

Communicates with the Halite engine via the standard hlt stdin/stdout protocol.
Each turn:
  1. Build a WorldView from the current game state.
  2. Extract scalar features + 9x9x6 map patch per ship.
  3. Run the network to get a desired action (intent) per ship.
  4. Pass intents through the deterministic resolver (no self-collisions).
  5. Decide spawn (rl_v5's economy rule) and issue all commands.

Reports the bot name as "rl_v7" so replays show "rl_v7" vs "rl_v5" correctly.

Usage:
    python rl_v7/rl_bot.py --model rl_v7/checkpoints/bc.pt
    python rl_v7/rl_bot.py --model rl_v7/checkpoints/bc.pt --deterministic
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE      = os.path.dirname(os.path.abspath(__file__))
MY_EXT    = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(MY_EXT)
sys.path.insert(0, HERE)
sys.path.insert(0, MY_EXT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'starter_kits', 'Python3'))

import hlt                                          # noqa: E402
from hlt.positionals import Direction               # noqa: E402

from config import (                                # noqa: E402
    ACTION_STAY, ACTION_NORTH, ACTION_EAST, ACTION_SOUTH, ACTION_WEST,
    ACTION_TO_DIR, game_max_turns, target_dropoffs, spawn_econ_ok,
    ENDGAME_BUFFER,
)
from net import ActorCritic                         # noqa: E402
import features as featmod                          # noqa: E402
from resolver import resolve                        # noqa: E402

_DIR_OBJ = {
    'n': Direction.North, 's': Direction.South,
    'e': Direction.East,  'w': Direction.West,
    'o': Direction.Still,
}


def _want_spawn(wv) -> bool:
    n_ships = len(wv.my_ships)
    num_dropoffs = len(wv.my_deposits) - 1
    tgt = target_dropoffs(wv.W, wv.H)
    turns_left = wv.max_turns - wv.turn
    return spawn_econ_ok(wv.my_bank, n_ships, turns_left, num_dropoffs, tgt)


def main(model_path: str, device_str: str = 'cpu', deterministic: bool = False):
    device = device_str

    model = ActorCritic.load(model_path, device=device)
    model.eval()

    game = hlt.Game()
    game.ready("rl_v7")

    while True:
        game.update_frame()
        me = game.me

        wv = featmod.world_view_from_hlt(game, me)

        # per-ship inference
        intents: dict = {}
        for ship in me.get_ships():
            sid = ship.id
            scal = featmod.extract_scalars(wv, sid)
            patch = featmod.extract_patch(wv, sid)
            mask = featmod.action_mask(wv, sid)
            if deterministic:
                act = model.greedy_action(scal, patch, mask=mask, device=device)
            else:
                act, _, _ = model.select_action(scal, patch, mask=mask, device=device)
            intents[sid] = act

        want_spawn = _want_spawn(wv)
        final, spawn_issued, dropoff_sid = resolve(wv, intents, want_spawn)

        commands = []
        for ship in me.get_ships():
            sid = ship.id
            if sid == dropoff_sid:
                commands.append(ship.make_dropoff())
                continue
            act = final.get(sid, ACTION_STAY)
            dir_ch = ACTION_TO_DIR[act]
            if dir_ch == 'o':
                commands.append(ship.stay_still())
            else:
                commands.append(ship.move(_DIR_OBJ[dir_ch]))

        if spawn_issued:
            commands.append(me.shipyard.spawn())

        game.end_turn(commands)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='rl_v7 inference bot')
    default_model = os.path.join(HERE, 'checkpoints', 'bc.pt')
    ap.add_argument('--model', default=default_model)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()
    main(args.model, args.device, args.deterministic)
