#!/usr/bin/env python3
"""Quick in-simulator diagnostics for a checkpoint: self-play games, per-seat stats."""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import BatchSim  # noqa: E402
from obs import encode, mirror_flags, uncanon_actions  # noqa: E402
from model import load  # noqa: E402
from train import policy_step  # noqa: E402


def run(model, B, seed, spawn_explore=None, greedy=False, opp=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sim = BatchSim(B)
    sim.reset([seed * 1000 + b for b in range(B)])
    mirror = mirror_flags(sim.fac, 32)
    fc = np.array([[(sim.fac[b, p, 1], 31 - sim.fac[b, p, 0] if mirror[b, p] else sim.fac[b, p, 0])
                    for p in (0, 1)] for b in range(B)])
    st = {k: np.zeros(2) for k in ('spawn', 'spawn_crash', 'self_crash', 'enemy_crash', 'deposited', 'drops', 'moves', 'stays')}
    while not sim.done.all():
        obs, mask, spok = encode(sim.halite, sim.owner, sim.occ, sim.cargo, sim.bank, sim.turn,
                                 sim.initial_halite, sim.fac, mirror)
        act = np.zeros((B, 2, 32, 32), np.int64)
        spawn = np.zeros((B, 2), bool)
        for p in (0, 1):
            net = model if (opp is None or p == 0) else opp
            a, _, _, sa, _, _ = policy_step(net, obs[:, p], mask[:, p], fc[:, p, 0], fc[:, p, 1], greedy=greedy)
            if spawn_explore is not None:
                sa = (rng.random(B) < spawn_explore).astype(int)
            for b in range(B):
                act[b, p] = uncanon_actions(np.where(mask[b, p, 0], a[b], 0), mirror[b, p])
                spawn[b, p] = sa[b] == 1 and spok[b, p]
                occp = sim.occ[b, p]
                st['moves'][p] += ((act[b, p] >= 1) & (act[b, p] <= 4) & occp).sum()
                st['stays'][p] += ((act[b, p] == 0) & occp).sum()
        occ_before = sim.occ.copy()
        ev = sim.step(act, spawn)
        for p in (0, 1):
            st['spawn'][p] += (ev['spawned'][:, p] >= 0).sum()
            st['spawn_crash'][p] += ev['spawn_credit'][:, p].size and 0
        dead = ev['dead']
        for b in range(B):
            ids = np.nonzero(dead[b])[0]
            for s in ids:
                kh = ev['kill_hulls'][b, s]
                where = [p for p in (0, 1) if (np.isin(s, np.where(occ_before[b, p], 0, -9)) or True)]
                if kh > 0:
                    st['enemy_crash'][0] += 1
                else:
                    st['self_crash'][0] += 1
        st['drops'] += 0
    st['deposited'] = sim.deposited.sum(0) / B
    st['bank'] = sim.bank.mean(0)
    return st


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--envs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--spawn-explore', type=float, default=None)
    ap.add_argument('--greedy', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(1)
    net, ex = load(a.model)
    net = net.to(memory_format=torch.channels_last).eval()
    st = run(net, a.envs, a.seed, a.spawn_explore, a.greedy)
    print('iter', ex.get('iter'), {k: np.round(v / (a.envs if k in ('spawn', 'moves', 'stays', 'self_crash', 'enemy_crash') else 1), 1).tolist() for k, v in st.items()})
