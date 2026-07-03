#!/usr/bin/env python3
"""
rl_v9 / resolver.py  --  FRIENDLY-ONLY deterministic collision-safety layer.

Deliberate rl_v9 change vs rl_v8: the resolver no longer knows enemies exist.
rl_v8 forced any ship near an enemy to STAY (and even forced ships to flee
enemy-adjacent cells), which meant the policy could never learn to trade
ships, contest cells, or break out of multi-turn traffic jams caused by enemy
bots camping nearby -- exactly the situations the owner wants learned.
Enemy interactions (including deliberate ramming) are now entirely up to the
policy; the reward in rl_env.py prices the exchange.

What stays hand-coded (fleet hygiene only):
1. Affordability: a ship that cannot pay the move cost is forced to STAY.
2. Endgame force-home: with turns_left <= dist + ENDGAME_BUFFER, loaded ships
   are recalled along the least-cost path (cargo would otherwise be lost at
   game end).
3. Factory reservation when spawning.
4. The cascade: no two FRIENDLY ships ever land on the same cell -- except the
   endgame pile-on exemption (wrecking on your own structure banks the cargo).

Returns (final_actions, spawn_issued, dropoff_ship_id).
"""

from typing import Dict, List, Optional, Set, Tuple

from config import (
    ACTION_STAY, ACTION_NORTH, ACTION_EAST, ACTION_SOUTH, ACTION_WEST,
    ACTION_DROPOFF, ACTION_DELTA, MOVE_COST_RATIO, ENDGAME_COLLAPSE_TURNS,
    ENDGAME_BUFFER,
)
import features as featmod

_CARDINALS = [
    (0, -1, ACTION_NORTH),
    (1,  0, ACTION_EAST),
    (0,  1, ACTION_SOUTH),
    (-1, 0, ACTION_WEST),
]


def resolve(
    wv,                          # WorldView for this turn
    intents: Dict[int, int],     # ship_id -> desired action (0-5)
    want_spawn: bool,
) -> Tuple[Dict[int, int], bool, Optional[int]]:
    W, H = wv.W, wv.H
    turns_left = wv.max_turns - wv.turn
    endgame = turns_left < ENDGAME_COLLAPSE_TURNS

    # ----- pick one dropoff ship (first that wants DROPOFF and legally can) --
    dropoff_sid: Optional[int] = None
    for sid, act in intents.items():
        if act == ACTION_DROPOFF:
            mask = featmod.action_mask(wv, sid)
            if mask[ACTION_DROPOFF]:
                dropoff_sid = sid
                break

    ship_pos: Dict[int, Tuple[int, int]] = {}
    cargo_of: Dict[int, int] = {}
    for sid, (sx, sy, cargo) in wv.my_ships.items():
        ship_pos[sid] = (sx, sy)
        cargo_of[sid] = cargo

    desired: Dict[int, int] = {}
    for sid in wv.my_ships:
        if sid == dropoff_sid:
            continue
        act = intents.get(sid, ACTION_STAY)
        if act == ACTION_DROPOFF:
            act = ACTION_STAY          # not the chosen dropoff ship
        desired[sid] = act

    # ----- affordability -----
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
                desired[sid] = _home_step(sx, sy, wv)

    # ----- factory reservation for spawn -----
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

    # include dropoff ship as permanent stayer so the cascade protects its cell
    if dropoff_sid is not None:
        work = dict(desired)
        work[dropoff_sid] = ACTION_STAY
    else:
        work = desired

    # endgame committed ships are exempt from the cascade (pile-on banks cargo)
    committed: Set[int] = set()
    if endgame:
        for sid, act in work.items():
            if act != ACTION_STAY:
                dx, dy = ACTION_DELTA[act]
                sx, sy = ship_pos[sid]
                dest = ((sx + dx) % W, (sy + dy) % H)
                if dest in deposits:
                    committed.add(sid)

    # cascade: no two friendly ships to the same cell
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

    final = {sid: work[sid] for sid in desired}

    # spawn: issue only if no friendly ship ends on the factory this turn
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

def _mdist(x1, y1, x2, y2, W, H):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return min(dx, W - dx) + min(dy, H - dy)


def _home_step(sx, sy, wv) -> int:
    """Least-cost step toward the nearest deposit (Dijkstra from all deposits)."""
    import heapq
    W, H = wv.W, wv.H
    halite = wv.halite
    deposits = [(d[0], d[1]) for d in wv.my_deposits]
    STEP_PENALTY = 20

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
            w = halite.get((nx, ny), 0) // MOVE_COST_RATIO + STEP_PENALTY
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
