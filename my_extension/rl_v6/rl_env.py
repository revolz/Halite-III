"""
Pure Halite III environment for rl_v6 (player 0 = the learner).

This is the rl_v5 environment stripped to its essentials: the engine, the
PURE engine-side feature extractors (14ch + 29 base scalars), and the
deposited-anchored reward.  The crucial difference is the command builder:

    rl_v6 executes player 0's primitive actions VERBATIM.

There is NO FSM, NO logit prior, NO meta-action resolution, NO homing memory
and NO 4-phase collision avoidance for player 0 — if two of its ships pick the
same cell they collide, exactly as the deployed pure bot would.  Spawning is an
explicit argument to `step` (decided by the learned spawn head by the caller).

Opponents are driven by an external `opponent_command_fn(env, pid) -> cmd_str`
(see experts.FrozenBotDriver) or a built-in greedy/idle scripted policy, so they
play faithfully (collision-aware) — only the learner is held to pure inference.

Action space (6): 0 STAY, 1 N, 2 E, 3 S, 4 W, 5 DROPOFF.
"""

import math
import os
import random
import sys
from typing import Dict, Optional, Tuple

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MY_EXT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _MY_EXT)

from halite_engine import HaliteEngine, SHIP_COST, MAX_HALITE, MOVE_COST_RATIO
from rl_features import (
    extract_spatial_from_engine, extract_scalars_from_engine,
    dropoff_legal, target_dropoffs, game_max_turns,
    torus_dist, torus_delta, _nearest_deposit, overlay_committed,
    ACTION_TO_DIR, ACTION_STAY, ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST,
    DROPOFF_MIN_TURNS_LEFT,
)
from rl_config import N_SHIP_ACTIONS_V6, ACTION_DROPOFF_V6

# ── reward weights (deposited-anchored; carried from rl_v5) ──────────────────
# W_DEP raised 1.0→2.0 (2026-06-27): with W_DEP=W_SHAPE the cargo-based Φ drop on the
# deposit turn exactly cancelled the deposit reward, so the policy hoarded and never
# delivered.  W_DEP=2 makes the deposit turn net +cargo (gap W_DEP−W_SHAPE > 0).
W_DEP           = 2.0
W_SHAPE         = 1.0
W_DROPOFF_PROXY = 50.0
W_WIN           = 200.0
W_MARGIN        = 400.0
MARGIN_SCALE    = 3000.0
SAFETY_MARGIN   = 8.0
COLLISION_FIXED = 100.0
W_IDLE          = 1.0
IDLE_GAIN_MIN   = 5

# ── per-ship reward revision (2026-06-27): mining/movement + attributed collisions ──
# All tunable knobs; magnitudes are relative to W_DEP=1.0 so a rich-cell mine
# (~+250/turn) rivals a small deposit and an offence outweighs a greedy lunge.
W_MINE                = 0.2     # gentle dwell-to-collect nudge (2026-06-27: 1.0→0.2 —
                                # at 1.0 mining ~11k/ep dwarfed deposit ~2.2k so the policy
                                # hoarded/churned; per halite unit delivery is now 11x
                                # better to mine+deliver (0.2) vs hoard (0.2) → +W_DEP=2.0)
W_MOVE_COST           = 0.3     # penalty per unit halite burned moving (1.0->0.3 2026-06-28:
                                # huge move-cost made advantages destructive; burned halite is
                                # already implicitly penalised via lower deposit)
W_DEATH               = 100.0   # per-ship penalty when a ship is destroyed (kept MODEST:
                                # the blameless first ship in a wreck eats this too)
W_INTENDED_COLLISION  = 200.0   # PRIMARY collision teacher: only the SECOND ship to pick an
                                # already-claimed cell (an "offender") pays it
W_IDLE_EMPTY          = 0.2     # softened idle: only a stay on a poor cell with empty cargo
IDLE_EMPTY_CARGO      = 50      # "empty" cargo threshold for the idle penalty
IDLE_POOR_CELL        = 50      # "poor" cell-halite threshold for the idle penalty

_DIR_DELTA = {ACTION_STAY: (0, 0), ACTION_NORTH: (0, -1), ACTION_SOUTH: (0, 1),
              ACTION_EAST: (1, 0), ACTION_WEST: (-1, 0)}


class HaliteEnvV6:
    """Single-agent (player 0) pure environment.

    obs    : {ship_id: (spatial[W,W,14], scalars[29], mask[6])}
    action : {ship_id: int in [0,6)}
    """

    def __init__(self, width=32, height=32, num_players=2, seed=None,
                 opponent_policy='greedy', allow_dropoff=True):
        self.width = width
        self.height = height
        self.num_players = num_players
        self._seed = seed
        self.opponent_policy = opponent_policy
        self.allow_dropoff = allow_dropoff
        self.opponent_command_fn = None   # callable(env, pid) -> cmd str
        self.engine: Optional[HaliteEngine] = None

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None):
        s = seed if seed is not None else self._seed
        self.engine = HaliteEngine(self.width, self.height, self.num_players,
                                   seed=s, verbose=False)
        self.engine._init_map_and_players()
        self.engine.turn = 0
        self._prev_deposited = {pid: 0 for pid in self.engine.players}
        self._prev_phi_ship = {}
        self.engine._update_inspiration()
        return self._get_obs(), {'turn': 0, 'max_turns': self.engine.max_turns}

    # --------------------------------------------------- sequential-decode obs
    def obs_for_ship(self, sid, committed_dests, vacated_origins):
        """Build one ship's observation with the Flavor-A overlay applied, so a
        ship sees the cells teammates have ALREADY committed to this turn.  Mirrors
        rl_bot.py's inference decode (train == inference)."""
        eng = self.engine
        sx, sy = eng.player_entities[0][sid]
        sp = extract_spatial_from_engine(eng, sid, 0)
        overlay_committed(sp, sx, sy, committed_dests, vacated_origins,
                          eng.width, eng.height)
        sc = extract_scalars_from_engine(eng, sid, 0)
        return sp, sc, self._mask(sid, 0)

    # ------------------------------------------------------------------- step
    def step(self, ship_actions: Dict[int, int], spawn: bool = False,
             offenders=None):
        eng = self.engine
        offenders = offenders or set()   # ships that chose an already-claimed cell
        eng.turn += 1
        eng._current_events = []
        eng.changed_cells.clear()
        eng._moved_entities.clear()

        pre_owners = {eid: eng.entities[eid]['owner'] for eid in eng.entities}
        pre_p0_cargo = {sid: eng.entities[sid]['cargo']
                        for sid, o in pre_owners.items() if o == 0}
        pre_pos    = {sid: eng.player_entities[0][sid] for sid in pre_p0_cargo}
        pre_cell_h = {sid: eng.halite.get(pre_pos[sid], 0) for sid in pre_p0_cargo}
        dropoffs_before = len(eng.players[0]['dropoffs'])
        dropped_before = eng._total_dropped.get(0, 0)

        p0_cmd, enacted_stays, dropoff_sid = self._build_p0_commands(ship_actions, spawn)
        converted_cargo = 0.0
        if dropoff_sid is not None and dropoff_sid in eng.entities:
            converted_cargo = eng.entities[dropoff_sid]['cargo']

        all_cmds = {0: p0_cmd}
        for pid in range(1, eng.num_players):
            all_cmds[pid] = self._opponent_command(pid)

        eng._process_commands(all_cmds)
        eng._process_mining()

        for pid in eng.players:
            if eng.player_entities[pid] or eng.players[pid]['energy'] >= SHIP_COST:
                eng._last_turn_alive[pid] = eng.turn
            n = len(eng.player_entities[pid])
            if n > eng._ships_peak[pid]:
                eng._ships_peak[pid] = n

        done = (eng.turn >= eng.max_turns) or eng._game_ended()

        # ---- PER-SHIP reward (2026-06-27 revision) ----------------------------
        # Each ship is credited/charged for its OWN actions so the gradient reaches
        # the responsible ship (the old fleet-average smeared every signal).
        deposited_now = eng._total_deposited.get(0, 0)
        deposited_this_turn = float(deposited_now - self._prev_deposited[0])
        self._prev_deposited[0] = deposited_now

        survivors = set(eng.player_entities[0].keys())
        dropoffs_after = len(eng.players[0]['dropoffs'])
        built_dropoff = dropoffs_after > dropoffs_before
        destroyed_p0 = set(pre_p0_cargo) - survivors
        if built_dropoff and dropoff_sid in destroyed_p0:
            destroyed_p0.discard(dropoff_sid)

        deposits_set = {eng.players[0]['factory']}
        deposits_set.update((dx, dy) for _d, dx, dy in eng.players[0]['dropoffs'])

        sr = {sid: 0.0 for sid in pre_p0_cargo}
        mined_total = 0.0
        move_cost_total = 0.0
        for sid in pre_p0_cargo:
            if sid in survivors:
                cur        = eng.player_entities[0][sid]
                cargo_now  = eng.entities[sid]['cargo']
                moved      = cur != pre_pos[sid]
                if moved:
                    mc = pre_cell_h[sid] // MOVE_COST_RATIO
                    sr[sid] -= W_MOVE_COST * mc
                    move_cost_total += mc
                else:
                    gain = cargo_now - pre_p0_cargo[sid]
                    if gain > 0 and cur not in deposits_set:
                        sr[sid] += W_MINE * gain          # dwell-and-mine reward
                        mined_total += gain
                    elif (cargo_now < IDLE_EMPTY_CARGO and
                          eng.halite.get(cur, 0) < IDLE_POOR_CELL):
                        sr[sid] -= W_IDLE_EMPTY           # genuinely idle on a poor cell
            else:
                sr[sid] -= W_DEATH                        # destroyed (modest; see weights)
            if sid in offenders:
                sr[sid] -= W_INTENDED_COLLISION           # second chooser of a taken cell

        # Deposit: split this turn's banked halite across ships sitting on a deposit.
        depositors = [sid for sid in survivors
                      if eng.player_entities[0][sid] in deposits_set
                      and pre_p0_cargo.get(sid, 0) > 0]
        if depositors and deposited_this_turn > 0:
            share = deposited_this_turn / len(depositors)
            for sid in depositors:
                sr[sid] += W_DEP * share

        # Potential-based shaping, decomposed per ship (Φ = Σ φ_sid).
        phi_ship = {} if done else self._phi_per_ship(0)
        for sid in pre_p0_cargo:
            sr[sid] += W_SHAPE * (phi_ship.get(sid, 0.0)
                                  - self._prev_phi_ship.get(sid, 0.0))
        self._prev_phi_ship = phi_ship

        if built_dropoff and dropoff_sid in sr:
            sr[dropoff_sid] += W_DROPOFF_PROXY + W_DEP * converted_cargo

        if done and sr:
            dep0 = eng._total_deposited.get(0, 0)
            opp = sum(eng._total_deposited.get(p, 0) for p in eng.players if p != 0)
            margin = dep0 - opp
            term = (W_WIN * (1.0 if margin > 0 else (-1.0 if margin < 0 else 0.0))
                    + W_MARGIN * math.tanh(margin / MARGIN_SCALE))
            share = term / len(sr)
            for sid in sr:
                sr[sid] += share

        if not done:
            eng._update_inspiration()

        reward = float(sum(sr.values()))
        info = {'turn': eng.turn, 'max_turns': eng.max_turns,
                'bank_p0': eng.players[0]['energy'],
                'ships_p0': len(eng.player_entities[0]),
                'deposited_p0': deposited_now,
                'ship_rewards': sr,
                'mined': mined_total,
                'move_cost': move_cost_total,
                'collisions': len(destroyed_p0),
                'offences': len(offenders)}
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------- p0 command build
    def _build_p0_commands(self, ship_actions: Dict[int, int], spawn: bool):
        """PURE: execute each ship's primitive verbatim.  At most one DROPOFF/turn
        (engine processes one construct cleanly); no collision/homing logic."""
        eng = self.engine
        tokens = []
        dropoff_sid = None
        enacted_stays = set()

        for sid in list(eng.player_entities[0].keys()):
            a = ship_actions.get(sid, ACTION_STAY)
            if a == ACTION_DROPOFF_V6:
                if self.allow_dropoff and dropoff_sid is None and self._dropoff_ok(sid):
                    dropoff_sid = sid
                    tokens.append(f"c {sid}")
                    continue
                a = ACTION_STAY   # illegal/second dropoff → mine
            # move-affordability: engine ignores unaffordable moves (stay/mine)
            if a != ACTION_STAY:
                sx, sy = eng.player_entities[0][sid]
                if eng.entities[sid]['cargo'] < eng.halite.get((sx, sy), 0) // MOVE_COST_RATIO:
                    a = ACTION_STAY
            if a == ACTION_STAY:
                enacted_stays.add(sid)
            tokens.append(f"m {sid} {ACTION_TO_DIR[a]}")

        if spawn and eng.players[0]['energy'] >= SHIP_COST:
            tokens.append('g')
        return ' '.join(tokens), enacted_stays, dropoff_sid

    def _dropoff_ok(self, sid: int) -> bool:
        eng = self.engine
        sx, sy = eng.player_entities[0][sid]
        cell_h = eng.halite.get((sx, sy), 0)
        cargo = eng.entities[sid]['cargo']
        deposits = [eng.players[0]['factory']] + [
            (dx, dy) for _d, dx, dy in eng.players[0]['dropoffs']]
        dist = min(torus_dist(sx, sy, d[0], d[1], eng.width, eng.height) for d in deposits)
        turns_left = eng.max_turns - eng.turn
        cell_owned = (sx, sy) in [tuple(d) for d in deposits]
        return dropoff_legal(eng.players[0]['energy'], cell_h, cargo, dist,
                             len(eng.players[0]['dropoffs']), turns_left, cell_owned)

    # ------------------------------------------------------------- opponents
    def _opponent_command(self, pid: int) -> str:
        if self.opponent_command_fn is not None:
            return self.opponent_command_fn(self, pid)
        eng = self.engine
        if self.opponent_policy == 'idle':
            return ''
        tokens = []
        for sid, (sx, sy) in list(eng.player_entities[pid].items()):
            if self.opponent_policy == 'random':
                tokens.append(f"m {sid} {random.choice(list(ACTION_TO_DIR.values()))}")
            else:  # greedy
                tokens.append(f"m {sid} {self._greedy_dir(pid, sid, sx, sy)}")
        if eng.players[pid]['energy'] >= SHIP_COST and len(eng.player_entities[pid]) < 10:
            tokens.append('g')
        return ' '.join(tokens)

    def _greedy_dir(self, pid, sid, sx, sy):
        eng = self.engine
        cargo = eng.entities[sid]['cargo']
        fx, fy = eng.players[pid]['factory']
        turns_left = eng.max_turns - eng.turn
        dist_home = torus_dist(sx, sy, fx, fy, eng.width, eng.height)
        if cargo >= MAX_HALITE * 0.9 or (cargo > 0 and turns_left <= dist_home + 5):
            return self._dir_toward(sx, sy, fx, fy)
        cell_h = eng.halite.get((sx, sy), 0)
        if cell_h > MAX_HALITE * 0.1:
            return 'o'
        best_h, best_dir = cell_h, 'o'
        for d, (ddx, ddy) in [('n', (0, -1)), ('s', (0, 1)), ('e', (1, 0)), ('w', (-1, 0))]:
            h = eng.halite.get(((sx + ddx) % eng.width, (sy + ddy) % eng.height), 0)
            if h > best_h:
                best_h, best_dir = h, d
        return best_dir

    def _dir_toward(self, sx, sy, tx, ty):
        W, H = self.engine.width, self.engine.height
        dx, dy = torus_delta(sx, sy, tx, ty, W, H)
        if abs(dx) >= abs(dy):
            return 'e' if dx > 0 else ('w' if dx < 0 else 'o')
        return 's' if dy > 0 else ('n' if dy < 0 else 'o')

    # ----------------------------------------------------------- observation
    def _phi_per_ship(self, pid=0):
        """Potential per ship (cargo weighted by ability to return in time).
        Φ = Σ φ_sid, so this is just _compute_phi split out for per-ship shaping."""
        eng = self.engine
        W, H = eng.width, eng.height
        turns_left = eng.max_turns - eng.turn
        out = {}
        for sid, (sx, sy) in eng.player_entities[pid].items():
            cargo = eng.entities[sid]['cargo']
            if cargo <= 0:
                continue
            near = _nearest_deposit(sx, sy, eng, pid)
            dist = torus_dist(sx, sy, near[0], near[1], W, H)
            rp = max(0.0, min(1.0, (turns_left - dist) / SAFETY_MARGIN))
            out[sid] = cargo * rp
        return out

    def _compute_phi(self, pid=0):
        return sum(self._phi_per_ship(pid).values())

    def _mask(self, sid, pid=0):
        eng = self.engine
        sx, sy = eng.player_entities[pid][sid]
        cell_h = eng.halite.get((sx, sy), 0)
        cargo = eng.entities[sid]['cargo']
        deposits = [eng.players[pid]['factory']] + [
            (dx, dy) for _d, dx, dy in eng.players[pid]['dropoffs']]
        dist = min(torus_dist(sx, sy, d[0], d[1], eng.width, eng.height) for d in deposits)
        turns_left = eng.max_turns - eng.turn
        cell_owned = (sx, sy) in [tuple(d) for d in deposits]
        ok = self.allow_dropoff and dropoff_legal(
            eng.players[pid]['energy'], cell_h, cargo, dist,
            len(eng.players[pid]['dropoffs']), turns_left, cell_owned)
        m = np.ones(N_SHIP_ACTIONS_V6, dtype=bool)
        m[ACTION_DROPOFF_V6] = bool(ok)
        return m

    def _get_obs(self):
        eng = self.engine
        obs = {}
        for sid in eng.player_entities[0]:
            sp = extract_spatial_from_engine(eng, sid, 0)
            sc = extract_scalars_from_engine(eng, sid, 0)   # 29 base scalars
            obs[sid] = (sp, sc, self._mask(sid, 0))
        return obs

    def spawn_features(self, pid=0):
        """Global spawn feature vector for the learned spawn head."""
        from rl_config import spawn_global_features
        eng = self.engine
        my_bank = eng.players[pid]['energy']
        opp_bank = max((eng.players[p]['energy'] for p in eng.players if p != pid),
                       default=0)
        my_ships = len(eng.player_entities[pid])
        opp_ships = sum(len(eng.player_entities[p]) for p in eng.players if p != pid)
        map_h = sum(eng.halite.values())
        return spawn_global_features(eng.turn, eng.max_turns, my_bank, opp_bank,
                                     my_ships, opp_ships, map_h, eng.width, eng.height)

    def close(self):
        self.engine = None
