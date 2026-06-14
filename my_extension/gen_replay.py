#!/usr/bin/env python3
"""
gen_replay.py – Generate a Halite III replay using the internal Python engine.

Runs player 0 as a greedy bot and player 1 as a greedy bot (or rl_v3 bot if
a model checkpoint is provided).  Saves a .hlt replay file you can open with
replay_viewer.py.

Usage:
    # Two greedy bots (no training needed):
    python gen_replay.py --seed 42 --output replays/my_game.hlt

    # rl_v3 vs greedy (after training):
    python gen_replay.py --model rl_v3/checkpoints/model_final_weights.pt --output replays/rl_v3_game.hlt
"""

import argparse
import os
import random
import sys
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from halite_engine import HaliteEngine, SHIP_COST, MAX_HALITE, DIRECTIONS


def greedy_command(engine, pid: int) -> str:
    """Simple greedy policy: mine if cell is rich, else move to richest neighbour,
    return home when cargo full or time is short. Spawns up to 10 ships."""
    from halite_engine import MOVE_COST_RATIO
    tokens = []
    eng = engine
    W, H = eng.width, eng.height
    factory = eng.players[pid]['factory']
    turns_left = eng.max_turns - eng.turn

    for ship_id, (sx, sy) in list(eng.player_entities[pid].items()):
        cargo = eng.entities[ship_id]['cargo']
        dist_home = abs(sx - factory[0]) + abs(sy - factory[1])

        # Force home near end-game or when almost full
        if cargo >= MAX_HALITE * 0.9 or (cargo > 0 and turns_left <= dist_home + 5):
            dx = factory[0] - sx
            dy = factory[1] - sy
            if abs(dx) > W // 2:
                dx = dx - W if dx > 0 else dx + W
            if abs(dy) > H // 2:
                dy = dy - H if dy > 0 else dy + H
            if abs(dx) >= abs(dy):
                d = 'e' if dx > 0 else 'w'
            else:
                d = 's' if dy > 0 else 'n'
            tokens.append(f"m {ship_id} {d}")
            continue

        cell_h = eng.halite.get((sx, sy), 0)
        if cell_h >= MAX_HALITE * 0.1:
            tokens.append(f"m {ship_id} o")
            continue

        # Move to richest adjacent cell
        best_h, best_d = cell_h, 'o'
        for d, (ddx, ddy) in [('n', (0,-1)), ('s', (0,1)), ('e', (1,0)), ('w', (-1,0))]:
            nx, ny = (sx+ddx) % W, (sy+ddy) % H
            h = eng.halite.get((nx, ny), 0)
            if h > best_h:
                best_h, best_d = h, d
        tokens.append(f"m {ship_id} {best_d}")

    # Spawn
    if eng.players[pid]['energy'] >= SHIP_COST and len(eng.player_entities[pid]) < 10:
        fx, fy = factory
        occupied = set(eng.player_entities[pid].values())
        adj = {((fx+ddx)%W, (fy+ddy)%H) for ddx,ddy in [(0,-1),(0,1),(1,0),(-1,0)]}
        if not (occupied & (adj | {(fx, fy)})):
            tokens.append('g')

    return ' '.join(tokens)


def run_greedy_vs_greedy(width, height, seed, replay_file):
    eng = HaliteEngine(width=width, height=height, num_players=2, seed=seed, verbose=False)
    eng._init_map_and_players()

    print(f"Running greedy vs greedy  map={width}x{height}  seed={seed}  turns={eng.max_turns}")

    for turn in range(1, eng.max_turns + 1):
        eng.turn = turn
        eng._current_events = []
        eng.changed_cells.clear()
        eng._moved_entities.clear()

        cmds = {0: greedy_command(eng, 0), 1: greedy_command(eng, 1)}
        eng._process_commands(cmds)
        eng._process_mining()

        for pid in eng.players:
            if eng.player_entities[pid] or eng.players[pid]['energy'] >= SHIP_COST:
                eng._last_turn_alive[pid] = eng.turn
            n = len(eng.player_entities[pid])
            if n > eng._ships_peak[pid]:
                eng._ships_peak[pid] = n

        eng._update_inspiration()

        if turn % 50 == 0:
            scores = {pid: eng.players[pid]['energy'] for pid in eng.players}
            print(f"  Turn {turn:3d}  P0={scores[0]:5d}  P1={scores[1]:5d}  "
                  f"ships P0={len(eng.player_entities[0])} P1={len(eng.player_entities[1])}")

        if eng._game_ended():
            print(f"  Game ended at turn {turn}")
            break

    os.makedirs(os.path.dirname(os.path.abspath(replay_file)), exist_ok=True)
    names = {pid: f'Greedy{pid}' for pid in eng.players}
    os.makedirs(os.path.dirname(os.path.abspath(replay_file)), exist_ok=True)
    eng.write_replay(replay_file, player_names=names)

    results = sorted(
        [(pid, eng.players[pid]['energy']) for pid in eng.players],
        key=lambda x: -x[1]
    )
    print(f"\nFinal results:")
    for rank, (pid, halite) in enumerate(results, 1):
        print(f"  #{rank} Player {pid}: {halite:,} halite")
    print(f"\nReplay saved: {replay_file}")
    return replay_file


def run_rl_vs_greedy(model_path, width, height, seed, replay_file):
    """Run the rl_v3 bot (player 0) vs greedy (player 1) and save a replay."""
    sys.path.insert(0, os.path.join(_HERE, 'rl_v3'))
    import torch
    from rl_model import ActorCritic
    from rl_env import HaliteEnv
    from rl_features import (N_SHIP_ACTIONS, ACTION_STAY, ACTION_HOME,
                              ACTION_RANDOM, ACTION_PROSPECT, PROSPECT_RADIUS,
                              _nearest_deposit, _richest_in_prospect_window,
                              ACTION_TO_DIR, torus_dist, torus_delta)
    from rl_env import HOME_CARGO_THRESHOLD, ENDGAME_BUFFER, MAX_DROPOFFS, DROPOFF_COST
    from halite_engine import EXTRACT_RATIO

    model = ActorCritic()
    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    env = HaliteEnv(width=width, height=height, num_players=2, seed=seed,
                    opponent_policy='greedy')
    obs, _ = env.reset()
    eng = env.engine

    print(f"Running rl_v3 vs greedy  map={width}x{height}  seed={seed}  turns={eng.max_turns}")

    done = False
    turn = 0
    while not done:
        # spawn heuristic
        turns_left = eng.max_turns - eng.turn
        spawn = (len(obs) < 8 and eng.players[0]['energy'] >= 2000 and turns_left > 75)

        actions = {}
        for ship_id, (spatial, scalars) in obs.items():
            sp_t = torch.from_numpy(spatial)
            sc_t = torch.from_numpy(scalars)
            with torch.no_grad():
                action, _, _ = model.select_action(sp_t, sc_t)
            actions[ship_id] = action

        obs, reward, done, info = env.step(actions, spawn=spawn)
        turn = info['turn']

        if turn % 50 == 0:
            scores = {pid: eng.players[pid]['energy'] for pid in eng.players}
            print(f"  Turn {turn:3d}  P0(rl_v3)={scores[0]:5d}  P1(greedy)={scores[1]:5d}  "
                  f"ships P0={len(eng.player_entities[0])} P1={len(eng.player_entities[1])}")

    names = {0: 'RL_v3', 1: 'Greedy'}
    os.makedirs(os.path.dirname(os.path.abspath(replay_file)), exist_ok=True)
    eng.write_replay(replay_file, player_names=names)

    results = sorted(
        [(pid, eng.players[pid]['energy']) for pid in eng.players],
        key=lambda x: -x[1]
    )
    print(f"\nFinal results:")
    for rank, (pid, halite) in enumerate(results, 1):
        name = "rl_v3" if pid == 0 else "greedy"
        print(f"  #{rank} Player {pid} ({name}): {halite:,} halite")
    winner = "rl_v3" if results[0][0] == 0 else "greedy"
    print(f"  Winner: {winner}!")
    print(f"\nReplay saved: {replay_file}")
    return replay_file


def main():
    parser = argparse.ArgumentParser(description='Generate a Halite III .hlt replay file')
    parser.add_argument('--model', default=None,
                        help='Path to rl_v3 model weights .pt file. If omitted, runs greedy vs greedy.')
    parser.add_argument('--width',  type=int, default=32)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--seed',   type=int, default=None)
    parser.add_argument('--output', default=None,
                        help='Output .hlt path. Auto-named if omitted.')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 99999)

    if args.output is None:
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tag = 'rl_v3_vs_greedy' if args.model else 'greedy_vs_greedy'
        out = os.path.join(_HERE, 'replays', f'replay-{ts}-{seed}-{tag}.hlt')
    else:
        out = args.output

    if args.model:
        run_rl_vs_greedy(args.model, args.width, args.height, seed, out)
    else:
        run_greedy_vs_greedy(args.width, args.height, seed, out)

    print(f"\nView with:")
    print(f"  C:\\Users\\PCTeo\\Miniconda3\\python.exe replay_viewer.py \"{out}\"")


if __name__ == '__main__':
    main()
