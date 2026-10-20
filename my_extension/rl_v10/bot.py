#!/usr/bin/env python3
"""
rl_v10 / bot.py -- live bot for halite_engine.py (and the standard Halite III
stdin/stdout protocol).  Parses the raw protocol into the same arrays the
simulator uses, encodes them with obs.encode, and samples every ship action
and the spawn decision from the network.  Nothing else decides anything.

Usage: python rl_v10/bot.py [--model rl_v10/checkpoints/best.pt] [--greedy | --sample]

Play mode: --greedy takes each ship's most likely action, --sample draws from the
network's probabilities.  With neither, the checkpoint's stored setting is used
(extra['greedy']; checkpoints without it sample).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from obs import encode, mirror_flags, uncanon_actions, factory_cell  # noqa: E402
from model import load  # noqa: E402

CMD = {1: 'n', 2: 'e', 3: 's', 4: 'w'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(HERE, 'checkpoints', 'best.pt'))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--greedy', dest='greedy', action='store_const', const=True, default=None)
    mode.add_argument('--sample', dest='greedy', action='store_const', const=False)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--dump', default=None, help='debug: save per-turn observations (.npz)')
    ap.add_argument('--name', default='rl_v10', help='name announced to the engine')
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    from train import policy_step
    net, extra = load(args.model)
    if args.greedy is None:
        args.greedy = bool(extra.get('greedy', False))
    net = net.to(memory_format=torch.channels_last).eval()

    rd = sys.stdin.readline
    json.loads(rd())
    n_players, me = map(int, rd().split())
    assert n_players == 2, 'rl_v10 plays 1v1'
    fac = np.zeros((1, 2, 2), np.int64)
    for _ in range(n_players):
        pid, fx, fy = map(int, rd().split())
        fac[0, pid] = (fx, fy)
    W, H = map(int, rd().split())
    halite = np.array([list(map(int, rd().split())) for _ in range(H)], np.int64)[None]
    initial = halite.sum((1, 2))
    owner = np.full((1, H, W), -1, np.int8)
    for p in range(2):
        owner[0, fac[0, p, 1], fac[0, p, 0]] = p
    mirror = mirror_flags(fac, W)
    fy, fx = factory_cell(fac[0, me], W, mirror[0, me])
    dumped = []
    sys.stdout.write(args.name + '\n')
    sys.stdout.flush()

    while True:
        line = rd()
        if not line:
            break
        turn = int(line)
        occ = np.zeros((1, 2, H, W), bool)
        cargo = np.zeros((1, 2, H, W), np.int64)
        sid = np.full((H, W), -1, np.int64)
        bank = np.zeros((1, 2), np.int64)
        for _ in range(n_players):
            pid, ns, nd, en = map(int, rd().split())
            bank[0, pid] = en
            for _ in range(ns):
                i, x, y, c = map(int, rd().split())
                occ[0, pid, y, x] = True
                cargo[0, pid, y, x] = c
                if pid == me:
                    sid[y, x] = i
            for _ in range(nd):
                _, x, y = map(int, rd().split())
                owner[0, y, x] = pid
        for _ in range(int(rd())):
            x, y, h = map(int, rd().split())
            halite[0, y, x] = h
        obs, mask, spok = encode(halite, owner, occ, cargo, bank, turn, initial, fac, mirror, seats=(me,))
        if args.dump:
            dumped.append(obs[0, 0].copy())
            np.save(args.dump, np.stack(dumped))
        a, _, _, sa, _, _ = policy_step(net, obs[:, 0], mask[:, 0], np.array([fy]), np.array([fx]),
                                         greedy=args.greedy)
        canon = np.where(mask[0, 0, 0], a[0], 0)
        act = uncanon_actions(canon, mirror[0, me])
        cmds = []
        for y, x in zip(*np.nonzero(occ[0, me])):
            k = int(act[y, x])
            if k == 5:
                cmds.append(f'c {sid[y, x]}')
            elif k in CMD:
                cmds.append(f'm {sid[y, x]} {CMD[k]}')
        if spok[0, 0] and sa[0] == 1:
            cmds.append('g')
        sys.stdout.write(' '.join(cmds) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
