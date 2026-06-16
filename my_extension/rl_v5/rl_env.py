"""
Gym-style Halite III environment for RL training.

The learning agent controls player 0; all other players use a scripted
opponent policy (random, greedy, or idle).

Collision prevention (applied every step before commands are sent)
------------------------------------------------------------------
Phase 1 – Build enemy threat zone: each enemy's cell + its 4 neighbours.
Phase 2 – Compute each ship's destination from its resolved primitive action.
Phase 3a – Ships whose destination is inside the threat zone are forced STAY.
Phase 3b – Ships already sitting at a threat-zone cell that are staying get an
           escape route: the first adjacent cell outside the threat zone.
Phase 4  – Friendly cascade: stayers own their cell; movers must yield.
           Heaviest mover wins when multiple movers compete for an empty cell.
           Iterated up to N times (N = number of ships) for convergence.

Spawn guard
-----------
A spawn command is suppressed if the factory cell or any of its 4 adjacent
cells is occupied by a friendly ship.

Home memory
-----------
A ship with cargo ≥ 60 % of MAX_HALITE is added to _homing_ships and stays
there until _home_dir() returns ACTION_STAY (ship arrived at deposit).

Usage
-----
    env = HaliteEnv(width=32, height=32)
    obs, info = env.reset()
    done = False
    while not done:
        # obs: {ship_id: (spatial[W,W,C], scalars[S], action_mask[N_ACTIONS])}
        spawn = len(obs) < 6 and env.engine.players[0]['energy'] >= 1000
        actions = {sid: random.randint(0, 4) for sid in obs}
        obs, reward, done, info = env.step(actions, spawn=spawn)
"""

import os
import sys
import math
import random
from typing import Dict, Optional, Tuple

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MY_EXT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)     # rl_v3/ — finds rl_features
sys.path.insert(0, _MY_EXT)  # my_extension/ — finds halite_engine

from halite_engine import (
    HaliteEngine,
    SHIP_COST, MAX_HALITE, DIRECTIONS, DROPOFF_COST, EXTRACT_RATIO,
)

# ── rl_v4 reward design (production-aligned) ────────────────────────────────
# v1 of this reward used a terminal bonus on raw BANK margin, which let the bot
# "win" by hoarding its 5000 starting capital (spawn one ship, camp the
# shipyard, deposit nothing) while the opponent spent its capital — a reward
# hack with deposited=0.  v2 anchors everything on PRODUCTION (halite actually
# deposited): own deposits dominate, the terminal bonus is small and based on
# DEPOSITED margin (not bank, so untouched starting capital is worthless), and a
# small idle penalty breaks the do-nothing local optimum / factory camping.
W_DEP            = 1.0      # reward per halite actually deposited (the objective)
W_SHAPE          = 1.0      # potential-based shaping weight (Φ in raw halite)
W_DROPOFF_PROXY  = 50.0     # one-time nudge to overcome the dropoff bank/cargo dip
W_WIN            = 200.0    # terminal: ± for out-depositing the opponent
W_MARGIN         = 400.0    # terminal: tanh(DEPOSITED margin) bonus (small vs production)
MARGIN_SCALE     = 3000.0   # halite scale for the terminal margin tanh
SAFETY_MARGIN    = 8.0      # turns of slack used in the cargo return-probability
COLLISION_FIXED  = 100.0    # fixed penalty per lost ship (≈ lost spawn investment)
W_IDLE           = 1.0      # per-ship penalty for staying without mining (anti-camp)
IDLE_GAIN_MIN    = 5        # cargo gain below which a STAY counts as idle
# rl_v5 §3 — move-cost penalty: DISABLED in the FSM-hybrid redesign.  The HARVEST
# state already keeps a ship mining a rich cell until it is (near) full, so an
# explicit move-cost penalty is redundant; worse, it penalised moving *toward* a
# rich cell (fighting the owner's "go to the richest cell" goal).  Kept as a tunable
# 0.0 rather than deleted so it is easy to re-enable for experiments.
W_MOVE_COST      = 0.0
# rl_v5 §2 — congestion penalty (anti-jam / dispersal).  Small and now applied
# ALL game (the owner saw end-game gridlock, not just opening jams); the FSM ESCAPE
# state does most of the dispersal, this just keeps production from clustering.
W_CONGEST        = 0.5      # per own-ship with another own-ship ≤1 step away

HOME_CARGO_THRESHOLD = 0.90  # return home when cargo >= 90% of MAX_HALITE
                             # (rl_v5: raised 0.75→0.90 to match the FSM HARVEST→HOME
                             #  "fully loaded" trigger — addresses "doesn't stay long
                             #  enough to harvest"; the builder's home-memory enforces it)
ENDGAME_BUFFER       = 5     # force home if turns_left <= dist_to_deposit + ENDGAME_BUFFER

from rl_features import (
    FSMController, FSMWorld, FSM_PRIOR_LOGIT, HARVEST, ESCAPE,
    extract_spatial_from_engine,
    extract_scalars_from_engine,
    fsm_feature_vector,
    fsm_prior_bonus,
    action_mask_from_engine,
    target_dropoffs,
    spawn_econ_ok,
    torus_dist,
    torus_delta,
    _nearest_deposit,
    _richest_in_prospect_window,
    compute_home_cost_field,
    least_cost_home_step,
    ENDGAME_COLLAPSE_TURNS,
    MOVE_COST_RATIO,
    ACTION_TO_DIR,
    ACTION_STAY,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_EAST,
    ACTION_WEST,
    ACTION_HOME,
    ACTION_RANDOM,
    ACTION_PROSPECT,
    ACTION_DROPOFF,
    N_SHIP_ACTIONS,
    PROSPECT_RADIUS,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HaliteEnv:
    """
    Single-agent (player 0) Halite III environment.

    Observation  {ship_id: (spatial float32[W,W,C], scalars float32[S],
                             action_mask bool[N_SHIP_ACTIONS])}
    Action       {ship_id: int in [0, N_SHIP_ACTIONS)}
                   0=Stay/Mine 1=N 2=E 3=S 4=W 5=RANDOM 6=HOME 7=PROSPECT 8=DROPOFF
    Reward       production-aligned (per step, raw-halite scale):
                   + W_DEP·deposited_this_turn          (the objective, dominant)
                   + W_SHAPE·(Φ_t − Φ_{t-1})            potential shaping
                   − (COLLISION_FIXED + cargo_lost)     per destroyed p0 ship
                   − W_IDLE·#idle_ships                 anti-camp / anti-do-nothing
                   + W_DROPOFF_PROXY + banked cargo     when a dropoff is built
                   + small terminal bonus on DEPOSITED margin vs the opponent
                     (NOT bank — so hoarding starting capital is worthless)
    Done         turn >= max_turns  or  engine._game_ended()
    """

    def __init__(
        self,
        width:              int  = 32,
        height:             int  = 32,
        num_players:        int  = 2,
        seed:               Optional[int] = None,
        opponent_policy:    str  = 'greedy',
        collision_scale:    float = 20.0,
        allow_dropoff:      bool = True,
    ):
        self.width           = width
        self.height          = height
        self.num_players     = num_players
        self._seed           = seed
        self.opponent_policy = opponent_policy
        self.collision_scale = collision_scale
        # When False, the DROPOFF action is masked out for player 0 (used by the
        # early curriculum phase to force pure mine→deposit learning).
        self.allow_dropoff   = allow_dropoff
        # Optional trained ActorCritic that drives ALL opponents (real self-play).
        # When None, opponents fall back to the scripted `opponent_policy`.
        self.opponent_model = None
        # Optional callable(env, pid) -> command string, used to drive opponents
        # whose policy/feature format differs from rl_v4 (e.g. the rl_v3 bot).
        # Takes precedence over opponent_model and the scripted policy.
        self.opponent_command_fn = None
        self.engine: Optional[HaliteEngine] = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        s = seed if seed is not None else self._seed
        self.engine = HaliteEngine(
            self.width, self.height, self.num_players,
            seed=s, verbose=False,
        )
        self.engine._init_map_and_players()
        self.engine.turn = 0

        # Track deposited halite across steps
        self._prev_deposited = {pid: 0 for pid in self.engine.players}
        # Ships currently committed to going home (memory mechanism)
        self._homing_ships: set = set()
        # Per-opponent home-memory sets (for model-driven self-play opponents)
        self._opp_homing: Dict[int, set] = {pid: set() for pid in self.engine.players}
        # Per-ship FSM controllers (rl_v5): one per player.  Player 0 is the agent;
        # the others are used only when a model drives a self-play opponent.
        self._fsm: Dict[int, FSMController] = {
            pid: FSMController() for pid in self.engine.players}
        # Last FSM-suggested action per p0 ship, cached when the obs was built so
        # the same action prior can be re-derived if needed.
        self._fsm_suggest: Dict[int, int] = {}
        # Potential Φ from the previous step (win-aligned reward shaping)
        self._prev_phi: float = 0.0

        # Pre-compute inspiration for turn-0 state (no ships → no-op but correct)
        self.engine._update_inspiration()

        obs  = self._get_obs()
        info = {'turn': 0, 'max_turns': self.engine.max_turns}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        ship_actions: Dict[int, int],
        spawn: bool = False,
    ) -> Tuple[dict, float, bool, dict]:
        """
        Execute one game turn.

        Inspiration has already been updated at the end of the previous step
        (or in reset), so we do NOT call it again at the top of this method.

        Returns
        -------
        obs     : {ship_id: (spatial, scalars)} for surviving player-0 ships
        reward  : float
        done    : bool
        info    : dict
        """
        eng        = self.engine
        eng.turn  += 1
        eng._current_events = []
        eng.changed_cells.clear()
        eng._moved_entities.clear()

        # Snapshot owner map BEFORE commands (destroyed ships are removed)
        pre_ship_owners = {eid: eng.entities[eid]['owner'] for eid in eng.entities}
        # Snapshot p0 cargo before commands for collision penalty calculation
        pre_p0_cargo = {sid: eng.entities[sid]['cargo']
                        for sid, owner in pre_ship_owners.items() if owner == 0}
        # Snapshot each p0 ship's move cost (rl_v5 §3): floor(cell halite / 10) at its
        # pre-move cell — the halite it would burn IF it moves this turn.
        pre_p0_movecost = {}
        pre_p0_pos = {}
        for sid in pre_p0_cargo:
            pos = eng.player_entities[0].get(sid)
            if pos is not None:
                pre_p0_pos[sid] = pos
                pre_p0_movecost[sid] = eng.halite.get(pos, 0) // MOVE_COST_RATIO

        # Snapshot dropoff count before commands
        dropoffs_before = len(eng.players[0]['dropoffs'])
        # Snapshot halite DROPPED (lost on collision) so the collision penalty counts
        # only cargo actually lost — a ship that "sacrifices" onto its own deposit
        # banks its cargo (credited to deposited above), so it must not be penalised.
        dropped_before = eng._total_dropped.get(0, 0)

        # Build commands
        p0_cmd, intent_actions, enacted_stays, dropoff_sid = \
            self._build_p0_commands(ship_actions, spawn)
        # Capture the converting ship's cargo BEFORE the construct consumes it,
        # so the reward can treat it as banked (the engine credits cargo+cell to
        # the player's energy on construct — it just isn't counted in deposits).
        converted_cargo = 0.0
        if dropoff_sid is not None and dropoff_sid in eng.entities:
            converted_cargo = eng.entities[dropoff_sid]['cargo']

        all_cmds  = {0: p0_cmd}
        for pid in range(1, eng.num_players):
            all_cmds[pid] = self._opponent_command(pid)

        eng._process_commands(all_cmds)
        eng._process_mining()

        # Per-player statistics
        for pid in eng.players:
            if eng.player_entities[pid] or eng.players[pid]['energy'] >= SHIP_COST:
                eng._last_turn_alive[pid] = eng.turn
            n = len(eng.player_entities[pid])
            if n > eng._ships_peak[pid]:
                eng._ships_peak[pid] = n

        # ------------------------------------------------------------------
        # Done (needed before reward: Φ(terminal) must be 0 for correct shaping)
        # ------------------------------------------------------------------
        done = (eng.turn >= eng.max_turns) or eng._game_ended()

        # ------------------------------------------------------------------
        # Production-aligned reward
        #   r = W_DEP·deposited                   own production (dominant term)
        #     + W_SHAPE·(Φ_t − Φ_{t-1})           potential shaping (raw halite)
        #     − (COLLISION_FIXED + cargo_lost)    per destroyed p0 ship
        #     − W_IDLE·#idle_ships                 anti-camp / anti-do-nothing
        #     + W_DROPOFF_PROXY (+ banked converted cargo) on a new dropoff
        #     + small terminal bonus on DEPOSITED margin vs the opponent
        # ------------------------------------------------------------------
        reward = 0.0

        # 1. Halite actually banked this turn — the objective.
        deposited_now = eng._total_deposited.get(0, 0)
        deposited_this_turn = float(deposited_now - self._prev_deposited[0])
        self._prev_deposited[0] = deposited_now
        reward += W_DEP * deposited_this_turn

        # 2. Potential-based shaping: cargo is valued only insofar as it can still
        #    be banked.  Φ(terminal)=0 so leftover cargo at game end is worthless.
        phi = 0.0 if done else self._compute_phi()
        reward += W_SHAPE * (phi - self._prev_phi)
        self._prev_phi = phi

        # 3. Collision penalty: fixed ship value per destroyed p0 ship + cargo ACTUALLY
        #    lost (dumped on collision).  Using the engine's dropped-halite delta means
        #    a ship that sacrifices onto its own deposit (cargo banked, not dumped) is
        #    not penalised for its cargo — only the ship-value term applies, so the
        #    endgame sacrifice is rewarded whenever cargo > COLLISION_FIXED.
        #    A ship that converted into a dropoff is intentionally removed — it is
        #    NOT a collision, so exclude it from the destroyed set.
        dropoffs_after  = len(eng.players[0]['dropoffs'])
        built_dropoff   = dropoffs_after > dropoffs_before
        p0_ships_before = set(pre_p0_cargo.keys())
        destroyed_p0    = p0_ships_before - set(eng.player_entities[0].keys())
        if built_dropoff and dropoff_sid in destroyed_p0:
            destroyed_p0.discard(dropoff_sid)
        dropped_this_turn = eng._total_dropped.get(0, 0) - dropped_before
        reward -= COLLISION_FIXED * len(destroyed_p0) + dropped_this_turn

        # 3b. Move-cost penalty (rl_v5 §3): the real halite burned by ships that
        #     MOVED this turn = Σ floor(cell halite / 10).  Makes leaving a rich cell
        #     costly so ships mine it down first, then leave cheaply (two birds).
        if W_MOVE_COST:
            move_cost_burned = sum(
                pre_p0_movecost.get(sid, 0)
                for sid in pre_p0_cargo
                if sid not in enacted_stays and sid != dropoff_sid
            )
            reward -= W_MOVE_COST * move_cost_burned

        # 4. Idle penalty: a ship that STAYED but mined essentially nothing is
        #    loafing (incl. camping the shipyard with empty cargo).  This breaks
        #    the do-nothing local optimum the bank-hoarding reward fell into.
        #    rl_v5: exclude ships committed to homing — a homing ship forced to STAY
        #    by collision resolution is not loafing, so must not be penalised.
        idle_ships = 0
        fsm_state = self._fsm[0].state
        for sid in enacted_stays:
            if sid in self._homing_ships:
                continue
            # A ship deliberately HARVESTing a rich cell or in ESCAPE is not loafing.
            if fsm_state.get(sid) in (HARVEST, ESCAPE):
                continue
            if sid in eng.entities and sid in pre_p0_cargo:
                if eng.entities[sid]['cargo'] - pre_p0_cargo[sid] < IDLE_GAIN_MIN:
                    idle_ships += 1
        reward -= W_IDLE * idle_ships

        # 4b. Congestion penalty (rl_v5 §2): penalise own ships bunched together
        #     (within 1 step) so the fleet spreads to the richer outer ring instead
        #     of jamming.  Applied all game (end-game gridlock was a problem) and kept
        #     small so it never competes with production; the FSM ESCAPE state is the
        #     primary dispersal mechanism.
        if W_CONGEST:
            positions = list(eng.player_entities[0].values())
            pos_set = set(positions)
            clustered = 0
            for (px, py) in positions:
                neigh = (((px + 1) % eng.width, py), ((px - 1) % eng.width, py),
                         (px, (py + 1) % eng.height), (px, (py - 1) % eng.height))
                if any(n in pos_set for n in neigh):
                    clustered += 1
            reward -= W_CONGEST * clustered

        # 5. Dropoff construction: small proxy bonus + bank the converted cargo
        #    (which Φ no longer counts but the engine credited to our energy).
        if built_dropoff:
            reward += W_DROPOFF_PROXY + W_DEP * converted_cargo

        # 6. Terminal bonus on DEPOSITED margin (NOT bank — hoarding the starting
        #    capital must not count).  Small so own production stays dominant.
        if done:
            dep0    = eng._total_deposited.get(0, 0)
            opp_dep = sum(eng._total_deposited.get(p, 0)
                          for p in eng.players if p != 0)
            margin  = dep0 - opp_dep
            reward += W_WIN * (1.0 if margin > 0 else (-1.0 if margin < 0 else 0.0))
            reward += W_MARGIN * math.tanh(margin / MARGIN_SCALE)

        # ------------------------------------------------------------------
        # Update inspiration for the NEXT step's observation
        # (mirrors engine behaviour: inspiration computed before frame is shown)
        # ------------------------------------------------------------------
        if not done:
            eng._update_inspiration()

        obs  = self._get_obs()
        info = {
            'turn':       eng.turn,
            'max_turns':  eng.max_turns,
            'bank_p0':    eng.players[0]['energy'],
            'ships_p0':   len(eng.player_entities[0]),
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    def _compute_phi(self, pid: int = 0) -> float:
        """Potential Φ = Σ cargo·return_prob over the player's ships (raw halite).

        return_prob ramps the value of a ship's cargo from 0 (cannot get home in
        time) to 1 (comfortably returnable), so the shaping reward only credits
        cargo that can plausibly be banked.
        """
        eng        = self.engine
        W, H       = eng.width, eng.height
        turns_left = eng.max_turns - eng.turn
        total = 0.0
        for sid, (sx, sy) in eng.player_entities[pid].items():
            cargo = eng.entities[sid]['cargo']
            if cargo <= 0:
                continue
            near = _nearest_deposit(sx, sy, eng, pid)
            dist = torus_dist(sx, sy, near[0], near[1], W, H)
            rp = max(0.0, min(1.0, (turns_left - dist) / SAFETY_MARGIN))
            total += cargo * rp
        return total

    def _build_p0_commands(self, ship_actions: Dict[int, int], spawn: bool):
        """Build the command string for player 0.

        Returns (cmd_str, intent_actions, enacted_stays, dropoff_ship_id).
        """
        return self._build_commands(0, ship_actions, spawn,
                                    self._homing_ships, self.allow_dropoff)

    def _build_commands(self, pid: int, ship_actions: Dict[int, int], spawn: bool,
                        homing_ships: set, allow_dropoff: bool):
        """Build a command string for `pid` from per-ship action indices.

        Handles the learned DROPOFF action (at most one construct per turn),
        home-memory, meta-action resolution, the 4-phase collision-avoidance
        safety layer, and spawning.  Used for player 0 and for model-driven
        self-play opponents alike.

        Returns (cmd_str, intent_actions, enacted_stays, dropoff_ship_id).
        """
        eng    = self.engine
        W, H   = eng.width, eng.height
        intent_actions: Dict[int, int] = {}
        resolved_prims: Dict[int, int] = {}

        # --- Pick at most one legal DROPOFF ship this turn (learned action) ---
        dropoff_ship_id = None
        for ship_id, action in ship_actions.items():
            if action == ACTION_DROPOFF and ship_id in eng.player_entities[pid]:
                mask = action_mask_from_engine(eng, ship_id, pid, allow_dropoff)
                if mask[ACTION_DROPOFF]:
                    dropoff_ship_id = ship_id
                    break

        for ship_id, action in ship_actions.items():
            if ship_id not in eng.player_entities[pid]:
                continue
            # Skip the ship being converted to a dropoff — it won't move.
            if ship_id == dropoff_ship_id:
                continue
            # A DROPOFF action that wasn't selected (illegal / not first) → STAY.
            if action == ACTION_DROPOFF:
                action = ACTION_STAY

            cargo = eng.entities[ship_id]['cargo']
            sx, sy = eng.player_entities[pid][ship_id]

            # Endgame force-home: ship mathematically cannot return in time
            turns_left = eng.max_turns - eng.turn
            near = _nearest_deposit(sx, sy, eng, pid)
            dist_dep = torus_dist(sx, sy, near[0], near[1], W, H)
            endgame = cargo > 0 and turns_left <= dist_dep + ENDGAME_BUFFER
            if endgame:
                homing_ships.add(ship_id)
            # rl_v5: a stuck homing ship the FSM put into ESCAPE breaks the shipyard
            # jam with a random move — release it from home-memory so that move
            # survives (never in endgame, where reaching home is mandatory).
            elif self._fsm[pid].state.get(ship_id) == ESCAPE:
                homing_ships.discard(ship_id)
            # Auto-trigger home if cargo exceeds threshold
            elif cargo >= MAX_HALITE * HOME_CARGO_THRESHOLD:
                homing_ships.add(ship_id)
            # Home memory: if this ship is committed to returning, keep it going
            if ship_id in homing_ships:
                action = ACTION_HOME
            intent_actions[ship_id] = action
            # Resolve meta-actions to primitive 0–4
            if action == ACTION_RANDOM:
                action = random.randint(0, 4)
            elif action == ACTION_HOME:
                homing_ships.add(ship_id)
                action = self._home_dir(ship_id, pid)
                if action == ACTION_STAY:   # arrived at deposit — cancel home mode
                    homing_ships.discard(ship_id)
            elif action == ACTION_PROSPECT:
                action = self._prospect_dir(ship_id, pid)
            resolved_prims[ship_id] = action

        # ── Safety override: 4-phase collision prevention ────────────────────
        _dir_delta = {
            ACTION_STAY: (0, 0), ACTION_NORTH: (0, -1), ACTION_SOUTH: (0, 1),
            ACTION_EAST: (1, 0), ACTION_WEST: (-1, 0),
        }
        # Cardinal-only deltas with their corresponding action index.
        _cardinal = [(0, -1, ACTION_NORTH), (1, 0, ACTION_EAST),
                     (0,  1, ACTION_SOUTH), (-1, 0, ACTION_WEST)]

        # Phase 1 — Build enemy threat zone.
        # Covers "enemy stays" AND "enemy moves adjacent": current + 4 neighbours.
        enemy_threat_zone: set = set()
        for other in range(eng.num_players):
            if other == pid:
                continue
            for _, epos in eng.player_entities[other].items():
                ex, ey = epos
                enemy_threat_zone.add((ex, ey))
                for ddx, ddy, _ in _cardinal:
                    enemy_threat_zone.add(((ex + ddx) % W, (ey + ddy) % H))

        # Phase 2 — Compute initial destinations (with move-affordability check).
        # The engine silently ignores any move a ship cannot afford
        # (cargo < cell_halite // 10).  Mirror this here so that planned
        # destinations reflect what the engine will actually execute.
        dest_map: Dict[int, tuple] = {}
        for sid, prim in resolved_prims.items():
            sx, sy = eng.player_entities[pid][sid]
            if prim != ACTION_STAY:
                cell_h = eng.halite.get((sx, sy), 0)
                if eng.entities[sid]['cargo'] < cell_h // 10:
                    prim = ACTION_STAY
                    resolved_prims[sid] = ACTION_STAY
            ddx, ddy = _dir_delta[prim]
            dest_map[sid] = ((sx + ddx) % W, (sy + ddy) % H)

        # Phase 3a — Enemy avoidance (MOVE): ships heading into threat zone → STAY.
        for sid in list(resolved_prims.keys()):
            if dest_map[sid] in enemy_threat_zone:
                resolved_prims[sid] = ACTION_STAY
                dest_map[sid] = eng.player_entities[pid][sid]

        # Phase 3b — Enemy avoidance (ESCAPE): a ship already at a threat-zone cell
        # while staying there risks collision if an adjacent enemy moves in.
        # Try to move to any safe adjacent cell instead of staying in the danger zone.
        for sid in list(resolved_prims.keys()):
            sx, sy = eng.player_entities[pid][sid]
            if resolved_prims[sid] == ACTION_STAY and (sx, sy) in enemy_threat_zone:
                for ddx, ddy, act in _cardinal:
                    esc = ((sx + ddx) % W, (sy + ddy) % H)
                    if esc not in enemy_threat_zone:
                        resolved_prims[sid] = act
                        dest_map[sid] = esc
                        break   # first safe direction wins; cascade handles friendlies

        # Phase 3c — Reserve the factory cell for a spawn (rl_v5).  If we intend to
        # spawn this turn, force any friendly heading onto the factory to STAY (off
        # it) so the new ship has the cell to itself.  Done BEFORE the Phase 4
        # cascade so any induced friendly conflicts are resolved consistently.
        fx0, fy0 = eng.players[pid]['factory']
        want_spawn = bool(spawn) and eng.players[pid]['energy'] >= SHIP_COST
        if want_spawn:
            for sid in list(resolved_prims.keys()):
                if dest_map[sid] == (fx0, fy0) and eng.player_entities[pid][sid] != (fx0, fy0):
                    resolved_prims[sid] = ACTION_STAY
                    dest_map[sid] = eng.player_entities[pid][sid]

        # Phase 4 — Friendly collision resolution: no two ships may share a final cell.
        # Rules (applied via cascade until stable):
        #   Stayer (dest == current_pos) owns that cell — every mover is forced STAY.
        #   Multiple movers to same empty cell → heaviest mover wins, rest STAY.
        # A forced STAY creates a new stayer, potentially blocking another ship;
        # iterating up to N times guarantees convergence for any chain of N ships.
        # rl_v5 endgame collapse: in the final ENDGAME_COLLAPSE_TURNS turns, ships
        # whose destination is one of our own deposit cells are exempt from the
        # cascade — they pile onto the shipyard/dropoff and the engine banks each
        # ship's cargo on its structure before the wreck (the "home sacrifice").
        deposit_cells = {eng.players[pid]['factory']}
        deposit_cells.update((dx, dy) for _did, dx, dy in eng.players[pid]['dropoffs'])
        endgame_collapse = (eng.max_turns - eng.turn) < ENDGAME_COLLAPSE_TURNS
        ship_list = [sid for sid in resolved_prims
                     if not (endgame_collapse and dest_map[sid] in deposit_cells)]
        for _ in range(len(ship_list)):
            pos_occupants: Dict[tuple, list] = {}
            for sid in ship_list:
                pos_occupants.setdefault(dest_map[sid], []).append(
                    (eng.entities[sid]['cargo'], sid)
                )
            changed = False
            for dest, occupants in pos_occupants.items():
                if len(occupants) <= 1:
                    continue
                stayers = [(c, sid) for c, sid in occupants
                           if dest == eng.player_entities[pid][sid]]
                movers  = [(c, sid) for c, sid in occupants
                           if dest != eng.player_entities[pid][sid]]
                if stayers:
                    for _, sid in movers:
                        resolved_prims[sid] = ACTION_STAY
                        dest_map[sid] = eng.player_entities[pid][sid]
                        changed = True
                elif len(movers) > 1:
                    movers.sort(reverse=True)
                    for _, sid in movers[1:]:
                        resolved_prims[sid] = ACTION_STAY
                        dest_map[sid] = eng.player_entities[pid][sid]
                        changed = True
            if not changed:
                break

        # Collect ships that ended up staying (for mining bonus computation)
        enacted_stays = {sid for sid, prim in resolved_prims.items() if prim == ACTION_STAY}

        tokens = [f"m {sid} {ACTION_TO_DIR[prim]}" for sid, prim in resolved_prims.items()]

        # Append the learned dropoff construct command (engine processes construct
        # before moves; the converting ship is already excluded from resolved_prims).
        if dropoff_ship_id is not None:
            tokens.append(f"c {dropoff_ship_id}")

        # Spawn guard (rl_v5): a spawn lands a ship ON the factory cell only, so the
        # single real constraint is that no friendly ENDS on the factory this turn.
        # rl_v4 also blocked spawning whenever a friendly merely sat on an *adjacent*
        # cell — far too conservative (we control our own ships; adjacency is safe).
        # The factory cell was reserved before the Phase 4 cascade (see want_spawn),
        # so here we just confirm it's free and emit the spawn.
        fx, fy = eng.players[pid]['factory']
        factory_cell_taken = any(dest_map[sid] == (fx, fy) for sid in resolved_prims)
        if want_spawn and not factory_cell_taken:
            tokens.append('g')
        return ' '.join(tokens), intent_actions, enacted_stays, dropoff_ship_id

    def _home_cost_field(self, pid: int) -> dict:
        """Least-cost-to-deposit field for `pid`, cached once per (turn, pid)."""
        eng = self.engine
        key = (eng.turn, pid)
        if getattr(self, '_hf_key', None) != key:
            deposits = [eng.players[pid]['factory']] + [
                (dx, dy) for _did, dx, dy in eng.players[pid]['dropoffs']]
            self._hf_field = compute_home_cost_field(
                eng.halite, deposits, eng.width, eng.height)
            self._hf_key = key
        return self._hf_field

    def _home_dir(self, ship_id: int, pid: int = 0) -> int:
        """Primitive action (0–4) toward the nearest deposit along the least-COST
        path (rl_v5; was pure Manhattan in rl_v4).  See least_cost_home_step."""
        eng    = self.engine
        sx, sy = eng.player_entities[pid][ship_id]
        field  = self._home_cost_field(pid)
        return least_cost_home_step(sx, sy, field, eng.width, eng.height)

    def _prospect_dir(self, ship_id: int, pid: int = 0) -> int:
        """Return the primitive action (0–4) that moves one step toward the
        richest halite cell in the PROSPECT window.  Returns ACTION_STAY when
        the ship is already on the local maximum (mine it)."""
        eng    = self.engine
        sx, sy = eng.player_entities[pid][ship_id]
        rx, ry, _ = _richest_in_prospect_window(
            sx, sy, eng.halite, eng.width, eng.height, PROSPECT_RADIUS,
        )
        if rx == sx and ry == sy:
            return ACTION_STAY
        dx, dy = torus_delta(sx, sy, rx, ry, eng.width, eng.height)
        if abs(dx) >= abs(dy):
            return ACTION_EAST if dx > 0 else ACTION_WEST
        else:
            return ACTION_NORTH if dy < 0 else ACTION_SOUTH

    def _model_opponent_command(self, pid: int) -> str:
        """Drive opponent `pid`'s fleet with self.opponent_model (real self-play).

        Uses the same observation/mask pipeline and command builder as player 0,
        so the opponent is a genuine copy of the policy under training.
        """
        import torch
        eng = self.engine
        actions: Dict[int, int] = {}
        # Same FSM-augmented observation + logit prior the agent uses, so a
        # self-play opponent is a genuine copy of the policy under training.
        opp_obs = self._obs_for_player(pid)
        for sid, (spatial, scalars, mask) in opp_obs.items():
            sp = torch.from_numpy(spatial)
            sc = torch.from_numpy(scalars)
            prior = fsm_prior_bonus(scalars, FSM_PRIOR_LOGIT)
            a, _, _ = self.opponent_model.select_action(sp, sc, mask=mask,
                                                        prior_bonus=prior)
            actions[sid] = a

        n_ships    = len(eng.player_entities[pid])
        turns_left = eng.max_turns - eng.turn
        tgt        = target_dropoffs(eng.width, eng.height)
        spawn      = spawn_econ_ok(eng.players[pid]['energy'], n_ships, turns_left,
                                   len(eng.players[pid]['dropoffs']), tgt)
        homing     = self._opp_homing.setdefault(pid, set())
        cmd, _, _, _ = self._build_commands(pid, actions, spawn, homing,
                                            allow_dropoff=True)
        return cmd

    def _opponent_command(self, pid: int) -> str:
        # External opponent (e.g. rl_v3) with its own policy/feature format.
        if self.opponent_command_fn is not None:
            return self.opponent_command_fn(self, pid)

        # Real self-play: a trained model drives the opponent's whole fleet.
        if self.opponent_model is not None:
            return self._model_opponent_command(pid)

        eng    = self.engine
        tokens = []

        if self.opponent_policy == 'idle':
            return ''

        for ship_id, (sx, sy) in list(eng.player_entities[pid].items()):
            if self.opponent_policy == 'random':
                direction = random.choice(list(ACTION_TO_DIR.values()))
                tokens.append(f"m {ship_id} {direction}")
            elif self.opponent_policy == 'greedy':
                tokens.append(f"m {ship_id} {self._greedy_dir(pid, ship_id, sx, sy)}")

        if eng.players[pid]['energy'] >= SHIP_COST and len(eng.player_entities[pid]) < 10:
            tokens.append('g')

        return ' '.join(tokens)

    def _greedy_dir(self, pid: int, ship_id: int, sx: int, sy: int) -> str:
        """Simple greedy: return home when full or time-critical, else mine."""
        eng        = self.engine
        cargo      = eng.entities[ship_id]['cargo']
        factory    = eng.players[pid]['factory']
        turns_left = eng.max_turns - eng.turn
        dist_home  = torus_dist(sx, sy, factory[0], factory[1], eng.width, eng.height)

        if cargo >= MAX_HALITE * 0.9 or (cargo > 0 and turns_left <= dist_home + 5):
            return self._dir_toward(sx, sy, factory[0], factory[1])

        cell_h = eng.halite.get((sx, sy), 0)
        if cell_h > MAX_HALITE * 0.1:
            return 'o'  # stay and mine

        best_h, best_dir = cell_h, 'o'
        for d, (ddx, ddy) in [('n', (0, -1)), ('s', (0, 1)), ('e', (1, 0)), ('w', (-1, 0))]:
            nx = (sx + ddx) % eng.width
            ny = (sy + ddy) % eng.height
            h  = eng.halite.get((nx, ny), 0)
            if h > best_h:
                best_h, best_dir = h, d
        return best_dir

    def _dir_toward(self, sx: int, sy: int, tx: int, ty: int) -> str:
        W, H = self.engine.width, self.engine.height
        dx   = tx - sx
        dy   = ty - sy
        if abs(dx) > W // 2:
            dx = dx - W if dx > 0 else dx + W
        if abs(dy) > H // 2:
            dy = dy - H if dy > 0 else dy + H
        if abs(dx) >= abs(dy):
            return 'e' if dx > 0 else ('w' if dx < 0 else 'o')
        return 's' if dy > 0 else ('n' if dy < 0 else 'o')

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _fsm_world(self, pid: int) -> FSMWorld:
        """Build the turn-global view the FSM needs for player `pid`."""
        eng = self.engine
        deposits = [eng.players[pid]['factory']] + [
            (dx, dy) for _did, dx, dy in eng.players[pid]['dropoffs']]
        friendly_pos = set(eng.player_entities[pid].values())
        return FSMWorld(
            halite=eng.halite,
            deposits=deposits,
            home_cost_field=self._home_cost_field(pid),
            friendly_pos=friendly_pos,
            W=eng.width, H=eng.height,
            turns_left=eng.max_turns - eng.turn,
        )

    def _obs_for_player(self, pid: int) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Observation per ship for `pid`, with the FSM feature block appended.

        Advances that player's FSM controller exactly once per turn (this method
        is the single per-turn obs builder for the player).
        """
        eng = self.engine
        fsm = self._fsm[pid]
        world = self._fsm_world(pid)
        fsm.begin_turn(eng.player_entities[pid].keys())
        suggest: Dict[int, int] = {}
        obs = {}
        for ship_id, (sx, sy) in eng.player_entities[pid].items():
            cargo  = eng.entities[ship_id]['cargo']
            cell_h = eng.halite.get((sx, sy), 0)
            action, state = fsm.decide(ship_id, sx, sy, cargo, cell_h, world)
            suggest[ship_id] = action

            spatial = extract_spatial_from_engine(eng, ship_id, pid)
            base_sc = extract_scalars_from_engine(eng, ship_id, pid)
            scalars = np.concatenate([base_sc, fsm_feature_vector(state, action)])
            mask    = action_mask_from_engine(eng, ship_id, pid,
                                              allow_dropoff=self.allow_dropoff)
            obs[ship_id] = (spatial, scalars, mask)
        if pid == 0:
            self._fsm_suggest = suggest
        return obs

    def _get_obs(self) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Observation per player-0 ship: (spatial, scalars, action_mask)."""
        return self._obs_for_player(0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def close(self):
        self.engine = None

    def get_action_meanings(self):
        return ['Stay/Mine', 'North', 'East', 'South', 'West']
