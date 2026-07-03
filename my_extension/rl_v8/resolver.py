#!/usr/bin/env python3
"""
rl_v8 / resolver.py  --  deterministic collision-safety layer.

The neural net outputs a *desired* primitive action (STAY/N/E/S/W/DROPOFF) for
each ship independently.  This module converts that fleet of intents into a
conflict-free set of final commands.  No two friendly ships will share a cell
after resolution.

Resolution algorithm (same spirit as rl_v5's Phase 4 cascade, but simpler
because rl_v7 has no meta-actions to resolve first):

1. Remove any desired move the ship cannot afford
   (cargo < floor(cell_halite / MOVE_COST_RATIO)).
2. Identify any enemy-threat cells (each enemy's current cell + 4 neighbours)
   and force ships heading there to STAY.
3. Reserve the factory cell if we want to spawn this turn.
4. Cascade until stable:
   - Stayer owns its cell; any mover targeting it is forced STAY.
   - Multiple movers to the same free cell: heaviest cargo wins, rest STAY.
5. Endgame collapse exemption (< ENDGAME_COLLAPSE_TURNS): ships heading to an
   own deposit cell are allowed to pile on (engine banks cargo before collision).

Returns the final (ship_id -> action_index) dict, a spawn_flag bool, and
optionally the dropoff_ship_id (or None).
"""

from typing import Dict, List, Optional, Set, Tuple

from config import (
    ACTION_STAY, ACTION_NORTH, ACTION_EAST, ACTION_SOUTH, ACTION_WEST,
    ACTION_DROPOFF, ACTION_DELTA, MOVE_COST_RATIO, ENDGAME_COLLAPSE_TURNS,
    HOME_CARGO_THRESHOLD, ENDGAME_BUFFER,
)
import features as featmod

_CARDINALS = [
    (0, -1, ACTION_NORTH),
    (1,  0, ACTION_EAST),
    (0,  1, ACTION_SOUTH),
    (-1, 0, ACTION_WEST),
]


def resolve(
    wv,              # WorldView for this turn
    intents: Dict[int, int],   # ship_id -> desired action (0-5)
    want_spawn: bool,
) -> Tuple[Dict[int, int], bool, Optional[int]]:
    """Resolve intents into collision-free actions.

    Returns (final_actions, spawn_issued, dropoff_ship_id).
    spawn_issued may be False even if want_spawn=True if the factory remains
    occupied after cascade.  dropoff_ship_id is None if no dropoff this turn.
    """
    W, H = wv.W, wv.H
    turns_left = wv.max_turns - wv.turn
    endgame = turns_left < ENDGAME_COLLAPSE_TURNS

    # ----- pick one dropoff ship (first that wants DROPOFF) -----
    dropoff_sid: Optional[int] = None
    for sid, act in intents.items():
        if act == ACTION_DROPOFF:
            # verify legality via action_mask
            mask = featmod.action_mask(wv, sid)
            if mask[ACTION_DROPOFF]:
                dropoff_sid = sid
                break

    # ----- build working set: non-dropoff ships -----
    ship_pos: Dict[int, Tuple[int, int]] = {}
    cargo_of: Dict[int, int] = {}
    for sid, (sx, sy, cargo) in wv.my_ships.items():
        ship_pos[sid] = (sx, sy)
        cargo_of[sid] = cargo

    desired: Dict[int, int] = {}   # sid -> action (will be mutated)
    for sid in wv.my_ships:
        if sid == dropoff_sid:
            continue
        act = intents.get(sid, ACTION_STAY)
        if act == ACTION_DROPOFF:
            act = ACTION_STAY   # not the chosen dropoff ship
        desired[sid] = act

    # ----- affordability: can't move if cargo < cell_halite // 10 -----
    for sid, act in desired.items():
        if act != ACTION_STAY:
            sx, sy = ship_pos[sid]
            cell_h = wv.halite.get((sx, sy), 0)
            if cargo_of[sid] < cell_h // MOVE_COST_RATIO:
                desired[sid] = ACTION_STAY

    # ----- endgame force-home -----
    deposits = set(tuple(d) for d in wv.my_deposits)
    for sid, act in desired.items():
        sx, sy = ship_pos[sid]
        if cargo_of[sid] > 0:
            dist = min(
                _mdist(sx, sy, dx, dy, W, H) for (dx, dy) in wv.my_deposits
            )
            if turns_left <= dist + ENDGAME_BUFFER:
                # home step
                desired[sid] = _home_step(sx, sy, wv)

    # ----- enemy threat zone -----
    threat: Set[Tuple[int, int]] = set()
    for (ex, ey, _) in wv.opp_ships:
        threat.add((ex, ey))
        for ddx, ddy, _ in _CARDINALS:
            threat.add(((ex + ddx) % W, (ey + ddy) % H))

    # Phase 3a: ships moving INTO threat zone -> STAY
    for sid, act in desired.items():
        if act != ACTION_STAY:
            dx, dy = ACTION_DELTA[act]
            sx, sy = ship_pos[sid]
            dest = ((sx + dx) % W, (sy + dy) % H)
            if dest in threat:
                desired[sid] = ACTION_STAY

    # Phase 3b: ships sitting IN a threat zone cell try to escape sideways
    for sid, act in desired.items():
        if act == ACTION_STAY:
            sx, sy = ship_pos[sid]
            if (sx, sy) in threat:
                for ddx, ddy, esc_act in _CARDINALS:
                    esc = ((sx + ddx) % W, (sy + ddy) % H)
                    if esc not in threat:
                        desired[sid] = esc_act
                        break

    # Phase 3c: factory reservation for spawn
    factory = wv.my_deposits[0]
    fxy = (factory[0], factory[1])
    if want_spawn:
        for sid, act in desired.items():
            if act != ACTION_STAY:
                dx, dy = ACTION_DELTA[act]
                sx, sy = ship_pos[sid]
                dest = ((sx + dx) % W, (sy + dy) % H)
                if dest == fxy and ship_pos[sid] != fxy:
                    desired[sid] = ACTION_STAY

    # include dropoff ship as permanent stayer so cascade protects its cell
    if dropoff_sid is not None:
        ds_xy = ship_pos[dropoff_sid]
        desired_with_dropoff = dict(desired)
        desired_with_dropoff[dropoff_sid] = ACTION_STAY
        work = desired_with_dropoff
    else:
        work = desired

    # endgame committed ships are exempt from cascade
    committed: Set[int] = set()
    if endgame:
        for sid, act in work.items():
            if act != ACTION_STAY:
                dx, dy = ACTION_DELTA[act]
                sx, sy = ship_pos[sid]
                dest = ((sx + dx) % W, (sy + dy) % H)
                if dest in deposits:
                    committed.add(sid)

    # Phase 4 cascade
    for _ in range(len(work) + 1):
        pos_occupants: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for sid, act in work.items():
            if sid in committed:
                continue
            dx, dy = ACTION_DELTA[act]
            sx, sy = ship_pos[sid]
            dest = ((sx + dx) % W, (sy + dy) % H)
            pos_occupants.setdefault(dest, []).append((cargo_of.get(sid, 0), sid))

        changed = False
        for dest, occupants in pos_occupants.items():
            if len(occupants) <= 1:
                continue
            stayers = [(c, s) for c, s in occupants if ship_pos[s] == dest]
            movers = [(c, s) for c, s in occupants if ship_pos[s] != dest]
            if stayers:
                for _, sid in movers:
                    if sid != dropoff_sid:
                        work[sid] = ACTION_STAY
                        changed = True
            elif len(movers) > 1:
                movers.sort(reverse=True)
                for _, sid in movers[1:]:
                    work[sid] = ACTION_STAY
                    changed = True
        if not changed:
            break

    # recover final desired (without dropoff ship's entry if it was added)
    final = {sid: work[sid] for sid in desired}

    # spawn: issue only if factory still free after cascade
    spawn_issued = False
    if want_spawn:
        factory_taken = False
        for sid, act in final.items():
            dx, dy = ACTION_DELTA[act]
            sx, sy = ship_pos[sid]
            dest = ((sx + dx) % W, (sy + dy) % H)
            if dest == fxy:
                factory_taken = True
                break
        spawn_issued = not factory_taken

    return final, spawn_issued, dropoff_sid


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mdist(x1, y1, x2, y2, W, H):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return min(dx, W - dx) + min(dy, H - dy)


def _home_step(sx, sy, wv) -> int:
    """Least-cost step toward nearest deposit (ported from rl_v5)."""
    import heapq
    W, H = wv.W, wv.H
    halite = wv.halite
    deposits = [(d[0], d[1]) for d in wv.my_deposits]
    MOVE_COST_RATIO_LOCAL = MOVE_COST_RATIO
    STEP_PENALTY = 20

    # Dijkstra from all deposits
    g: dict = {}
    pq = []
    for dx, dy in deposits:
        g[(dx, dy)] = 0
        heapq.heappush(pq, (0, dx, dy))
    while pq:
        cost, cx, cy = heapq.heappop(pq)
        if cost > g.get((cx, cy), float('inf')):
            continue
        for ddx, ddy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            nx, ny = (cx + ddx) % W, (cy + ddy) % H
            w = halite.get((nx, ny), 0) // MOVE_COST_RATIO_LOCAL + STEP_PENALTY
            nc = cost + w
            if nc < g.get((nx, ny), float('inf')):
                g[(nx, ny)] = nc
                heapq.heappush(pq, (nc, nx, ny))

    if g.get((sx, sy), -1) == 0:
        return ACTION_STAY
    best_act, best_g = ACTION_STAY, float('inf')
    for ddx, ddy, act in ((0, -1, ACTION_NORTH), (1, 0, ACTION_EAST),
                          (0, 1, ACTION_SOUTH), (-1, 0, ACTION_WEST)):
        n = ((sx + ddx) % W, (sy + ddy) % H)
        gv = g.get(n, float('inf'))
        if gv < best_g:
            best_g, best_act = gv, act
    return best_act
