#!/usr/bin/env python3
"""
rl_v9 / rl_bot.py  --  live inference bot (standard hlt stdin/stdout protocol).

Each turn:
  1. Build a WorldView; advance the FleetMemory (homing / prev-action / stuck).
  2. Extract per-ship features (scalars + 9x9x6 patch + 8x8x4 global) and run
     the ship policy for an intent per ship.
  3. Run the SpawnPolicy on the global features -- spawning is LEARNED, there
     is no fleet cap and no hand-coded spawn rule.
  4. Resolve friendly conflicts only (enemy collisions are the policy's call).
  5. Commit executed actions back into FleetMemory.

Usage:
    python rl_v9/rl_bot.py --model rl_v9/checkpoints/best.pt --deterministic
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
    ACTION_STAY, ACTION_DROPOFF, ACTION_TO_DIR, SPAWN_YES,
)
from net import load_bundle, NEG_INF                # noqa: E402
import features as featmod                          # noqa: E402
from features import FleetMemory                    # noqa: E402
from resolver import resolve                        # noqa: E402

_DIR_OBJ = {
    'n': Direction.North, 's': Direction.South,
    'e': Direction.East,  'w': Direction.West,
    'o': Direction.Still,
}


def main(model_path: str, device: str = 'cpu', deterministic: bool = False):
    policy, spawn_policy, _, _ = load_bundle(model_path, device=device)
    policy.eval()
    spawn_policy.eval()

    game = hlt.Game()
    game.ready("rl_v9")

    mem = FleetMemory()

    while True:
        game.update_frame()
        me = game.me
        wv = featmod.world_view_from_hlt(game, me)
        mem.begin_turn(wv)

        ships = me.get_ships()
        intents: dict = {}
        if ships:
            sids = [s.id for s in ships]
            scals = np.stack([featmod.extract_scalars(wv, s, mem) for s in sids])
            patches = np.stack([featmod.extract_patch(wv, s) for s in sids])
            gmaps = np.stack([featmod.extract_ship_global(wv, s) for s in sids])
            masks = np.stack([featmod.action_mask(wv, s) for s in sids])
            with torch.no_grad():
                st = torch.as_tensor(scals, dtype=torch.float32, device=device)
                pt = torch.as_tensor(np.transpose(patches, (0, 3, 1, 2)),
                                     dtype=torch.float32, device=device)
                gt = torch.as_tensor(gmaps, dtype=torch.float32, device=device)
                mt = torch.as_tensor(masks, dtype=torch.bool, device=device)
                logits = policy(st, pt, gt)
                logits = torch.where(mt, logits, torch.full_like(logits, NEG_INF))
                if deterministic:
                    acts = torch.argmax(logits, dim=1)
                else:
                    acts = torch.distributions.Categorical(logits=logits).sample()
            for i, sid in enumerate(sids):
                intents[sid] = int(acts[i].item())

        # learned spawn decision
        sscal = featmod.extract_spawn_scalars(wv)
        sglob = featmod.extract_spawn_global(wv)
        smask = featmod.spawn_mask(wv)
        with torch.no_grad():
            st = torch.as_tensor(sscal[None, ...], dtype=torch.float32, device=device)
            gt = torch.as_tensor(sglob[None, ...], dtype=torch.float32, device=device)
            slogits = spawn_policy(st, gt)[0]
            slogits = torch.where(torch.as_tensor(smask, device=device),
                                  slogits, torch.full_like(slogits, NEG_INF))
            if deterministic:
                s_act = int(torch.argmax(slogits).item())
            else:
                s_act = int(torch.distributions.Categorical(logits=slogits).sample().item())
        want_spawn = (s_act == SPAWN_YES)

        final, spawn_issued, dropoff_sid = resolve(wv, intents, want_spawn)
        mem.commit_actions(final)
        if dropoff_sid is not None:
            mem.commit_actions({dropoff_sid: ACTION_DROPOFF})

        commands = []
        for ship in ships:
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
    ap = argparse.ArgumentParser(description='rl_v9 inference bot')
    default_model = os.path.join(HERE, 'checkpoints', 'best.pt')
    ap.add_argument('--model', default=default_model)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()
    main(args.model, args.device, args.deterministic)
