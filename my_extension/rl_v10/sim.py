"""
rl_v10 / sim.py -- batched, vectorised re-implementation of halite_engine.py
for 2-player games.  Rules (including the engine's quirks) are copied from
halite_engine.py and checked turn-by-turn against real engine replays by
verify_sim.py.  B games advance in lock-step; all state is grid-based numpy.

Ship actions per cell: 0 stay, 1 N, 2 E, 3 S, 4 W, 5 construct dropoff.
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from halite_engine import generate_map  # noqa: E402

SHIP_COST = 1000
DROPOFF_COST = 4000
MAX_CARGO = 1000
INITIAL_ENERGY = 5000
MAX_TURNS = 400
INSP_RADIUS = 4
INSP_COUNT = 2
MAX_IDS = 2048

# (dy, dx) for actions 1..4
DIRS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
DIAMOND = [(dy, dx) for dy in range(-INSP_RADIUS, INSP_RADIUS + 1)
           for dx in range(-INSP_RADIUS, INSP_RADIUS + 1)
           if abs(dy) + abs(dx) <= INSP_RADIUS and (dy, dx) != (0, 0)]


def shift(a, dy, dx):
    """Value at (y, x) moves to (y+dy, x+dx) on the torus (last two axes)."""
    return np.roll(a, (dy, dx), axis=(-2, -1))


def near_count(occ):
    """Number of ships in occ within Manhattan radius 4 of each cell."""
    out = np.zeros(occ.shape, dtype=np.int16)
    o = occ.astype(np.int16)
    for dy, dx in DIAMOND:
        out += shift(o, dy, dx)
    return out


class BatchSim:
    def __init__(self, B, W=32, H=32):
        self.B, self.W, self.H = B, W, H

    # ------------------------------------------------------------------
    def reset(self, seeds, init_bank=None):
        B, H, W = self.B, self.H, self.W
        assert len(seeds) == B
        self.halite = np.zeros((B, H, W), np.int64)
        self.owner = np.full((B, H, W), -1, np.int8)
        self.struct_sid = np.full((B, H, W), -1, np.int64)
        self.fac = np.zeros((B, 2, 2), np.int64)          # (x, y)
        for b, s in enumerate(seeds):
            grid, facs = generate_map(W, H, 2, random.Random(s))
            self.halite[b] = np.array(grid, np.int64)
            for p, (fx, fy) in enumerate(facs):
                self.fac[b, p] = (fx, fy)
                self.owner[b, fy, fx] = p
                self.halite[b, fy, fx] = 0
        self.initial_halite = self.halite.sum(axis=(1, 2))
        self.occ = np.zeros((B, 2, H, W), bool)
        self.cargo = np.zeros((B, 2, H, W), np.int64)
        self.sid = np.full((B, 2, H, W), -1, np.int64)
        self.bank = np.full((B, 2), INITIAL_ENERGY, np.int64)
        if init_bank is not None:
            self.bank[:] = np.asarray(init_bank, np.int64)[:, None]
        self.next_id = np.zeros(B, np.int64)
        self.turn = 1
        self.done = np.zeros(B, bool)
        self.drop_list = [[] for _ in range(B)]      # (dropoff id, owner, x, y) in creation order
        self.next_drop_id = np.zeros(B, np.int64)
        self.deposited = np.zeros((B, 2), np.int64)
        self._update_insp()

    def _update_insp(self):
        self.near = np.stack([near_count(self.occ[:, 1]), near_count(self.occ[:, 0])], 1)
        self.insp = self.occ & (self.near >= INSP_COUNT)   # near[:, p] = enemy ships near

    # ------------------------------------------------------------------
    def step(self, act, spawn):
        """act: (B,2,H,W) int; spawn: (B,2) bool.  Returns per-ship event tables."""
        B, H, W = self.B, self.H, self.W
        act = np.where(self.occ, act, 0)
        act[self.done] = 0
        spawn = spawn & ~self.done[:, None]
        halite, owner, occ, cargo, sid, bank = self.halite, self.owner, self.occ, self.cargo, self.sid, self.bank

        ev = {
            'dep': np.zeros((B, MAX_IDS), np.int64),        # halite banked by ship
            'dead': np.zeros((B, MAX_IDS), bool),
            'kill_cargo': np.zeros((B, MAX_IDS), np.float64),
            'kill_hulls': np.zeros((B, MAX_IDS), np.float64),
            'construct_delta': np.zeros((B, MAX_IDS), np.int64),
            'constructed': np.zeros((B, MAX_IDS), bool),
            'struct_dep': np.zeros((B, MAX_IDS), np.int64),  # deposits at dropoff built by sid
            'spawned': np.full((B, 2), -1, np.int64),
            'spawn_credit': np.zeros((B, 2), np.int64),
        }
        bidx = np.arange(B)[:, None, None]

        pre_energy = bank.copy()
        expenses = np.zeros((B, 2), np.int64)

        # ---- 1. construct (row-major command order per player) ----
        req = occ & (act == 5) & (owner < 0)[:, None]
        if req.any():
            for b, p, y, x in zip(*np.nonzero(req)):
                credit = halite[b, y, x] + cargo[b, p, y, x]
                cost = max(0, DROPOFF_COST - credit)
                if expenses[b, p] + cost > pre_energy[b, p]:
                    continue
                expenses[b, p] += cost
                s = sid[b, p, y, x]
                owner[b, y, x] = p
                self.struct_sid[b, y, x] = s
                self.drop_list[b].append((int(self.next_drop_id[b]), p, int(x), int(y)))
                self.next_drop_id[b] += 1
                bank[b, p] += credit - cost
                ev['constructed'][b, s] = True
                ev['construct_delta'][b, s] = credit - cost
                ev['dep'][b, s] += cargo[b, p, y, x]
                halite[b, y, x] = 0
                occ[b, p, y, x] = False
                cargo[b, p, y, x] = 0
                sid[b, p, y, x] = -1
        act = np.where(occ, act, 0)

        # ---- 2. moves ----
        cost = (halite // 10)[:, None]
        mv = occ & (act >= 1) & (act <= 4)
        ok = mv & (cargo >= cost)
        cargo -= np.where(ok, cost, 0)
        stationary = occ & ~ok

        arr = np.zeros((B, 2, H, W), np.int16)
        moved_cargo = np.zeros((B, 2, H, W), np.int64)
        moved_sid = np.full((B, 2, H, W), -1, np.int64)
        for a, (dy, dx) in DIRS.items():
            m = ok & (act == a)
            if not m.any():
                continue
            arr += shift(m, dy, dx)
            moved_cargo += shift(np.where(m, cargo, 0), dy, dx)
            moved_sid = np.where(shift(m, dy, dx), shift(sid, dy, dx), moved_sid)
        arr_tot = arr.sum(1)
        contested = arr_tot >= 1
        total = arr_tot + (stationary.sum(1) * contested)
        crash = total >= 2                                    # (B,H,W)

        # ships dying: arriving movers at crash cells + stationary at crash cells
        if crash.any():
            die_mv = (arr > 0) & crash[:, None]
            die_st = stationary & crash[:, None]
            cell_n = die_mv * arr.astype(np.int64) + die_st          # ships per player per cell
            cell_c = np.where(die_mv, moved_cargo, 0) + np.where(die_st, cargo, 0)
            die_start = stationary & crash[:, None]
            for a, (dy, dx) in DIRS.items():
                die_start |= ok & (act == a) & shift(crash, -dy, -dx)[:, None]
            for b, p, y, x in zip(*np.nonzero(die_start)):
                a = act[b, p, y, x] if ok[b, p, y, x] else 0
                dy, dx = DIRS.get(int(a), (0, 0))
                ty, tx = (y + dy) % H, (x + dx) % W
                s = sid[b, p, y, x]
                c = cargo[b, p, y, x]
                q = 1 - p
                n_me = cell_n[b, p, ty, tx]
                ev['dead'][b, s] = True
                ev['kill_cargo'][b, s] += cell_c[b, q, ty, tx] / n_me
                ev['kill_hulls'][b, s] += cell_n[b, q, ty, tx] / n_me
                o = owner[b, ty, tx]
                if o == p:
                    bank[b, p] += c
                    self.deposited[b, p] += c
                    ev['dep'][b, s] += c
                    ss = self.struct_sid[b, ty, tx]
                    if ss >= 0:
                        ev['struct_dep'][b, ss] += c
                elif o >= 0:
                    if c > 0:
                        bank[b, o] += c
                elif c > 0:
                    halite[b, ty, tx] += c

        surv_mv = (arr > 0) & ~crash[:, None]
        surv_st = stationary & ~crash[:, None]
        new_occ = surv_mv | surv_st
        new_cargo = np.where(surv_mv, moved_cargo, np.where(surv_st, cargo, 0))
        new_sid = np.where(surv_mv, moved_sid, np.where(surv_st, sid, -1))
        nomine = surv_mv.copy()

        # ---- 3. dump on own structures ----
        for p in (0, 1):
            on = new_occ[:, p] & (owner == p)
            if on.any():
                amt = np.where(on, new_cargo[:, p], 0)
                bank[:, p] += amt.sum((1, 2))
                self.deposited[:, p] += amt.sum((1, 2))
                bb, yy, xx = np.nonzero(amt)
                ss = new_sid[bb, p, yy, xx]
                np.add.at(ev['dep'], (bb, ss), amt[bb, yy, xx])
                st = self.struct_sid[bb, yy, xx]
                k = st >= 0
                np.add.at(ev['struct_dep'], (bb[k], st[k]), amt[bb, yy, xx][k])
                new_cargo[:, p] = np.where(on, 0, new_cargo[:, p])
                nomine[:, p] |= on

        # ---- 4. spawn ----
        for p in (0, 1):
            for b in np.nonzero(spawn[:, p])[0]:
                if expenses[b, p] + SHIP_COST > pre_energy[b, p]:
                    continue
                fx, fy = self.fac[b, p]
                bank[b, p] -= SHIP_COST
                new = self.next_id[b]
                self.next_id[b] += 1
                ev['spawned'][b, p] = new
                if new_occ[b, :, fy, fx].any():
                    q = 0 if new_occ[b, 0, fy, fx] else 1
                    ex_c = new_cargo[b, q, fy, fx]
                    bank[b, p] += ex_c
                    ev['spawn_credit'][b, p] += ex_c
                    ev['dead'][b, new_sid[b, q, fy, fx]] = True
                    ev['dead'][b, new] = True
                    new_occ[b, q, fy, fx] = False
                    new_cargo[b, q, fy, fx] = 0
                    new_sid[b, q, fy, fx] = -1
                else:
                    new_occ[b, p, fy, fx] = True
                    new_cargo[b, p, fy, fx] = 0
                    new_sid[b, p, fy, fx] = new
                    nomine[b, p, fy, fx] = True

        # ---- 5. mining ----
        h = halite[:, None]
        can = new_occ & ~nomine & (new_cargo < MAX_CARGO) & (h > 0)
        ext = np.minimum(-(-h // 4), MAX_CARGO - new_cargo)
        ext = np.where(can, ext, 0)
        gain = np.where(self.insp, ext * 3, ext)
        gain = np.minimum(gain, MAX_CARGO - new_cargo)
        gain = np.where(can, gain, 0)
        new_cargo += gain
        halite -= ext.sum(1)

        self.occ, self.cargo, self.sid = new_occ, new_cargo, new_sid
        self.turn += 1
        alive = new_occ.any((2, 3)) | (bank >= SHIP_COST)
        self.done |= alive.sum(1) <= 1
        if self.turn > MAX_TURNS:
            self.done[:] = True
        self._update_insp()
        return ev

    def scores(self):
        return self.bank.copy()
