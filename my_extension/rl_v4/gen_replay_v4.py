#!/usr/bin/env python3
"""
gen_replay_v4.py – Generate a Halite III replay for the rl_v4 bot.

Two modes:

  1. rl_v4 (player 0) vs another RL bot (rl_v3 / rl_v2) — head-to-head over the
     hlt protocol so each bot uses its OWN feature/action module.  Use this to
     confirm rl_v4 builds dropoffs and out-deposits the archived bots:

         python gen_replay_v4.py --opponent-bot rl_v3 \
             --model rl_v4/checkpoints/model_final_weights.pt --seed 42

     (--model2 overrides the opponent's weights path; otherwise its own
      checkpoints/model_final_weights.pt is used.)

  2. rl_v4 vs greedy — runs in-process via HaliteEnv and writes a replay:

         python gen_replay_v4.py --model rl_v4/checkpoints/model_final_weights.pt --greedy

All paths are relative to the my_extension/ directory.
"""

import argparse
import datetime
import os
import random
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))   # rl_v4/
_MY_EXT = os.path.dirname(_HERE)                        # my_extension/
sys.path.insert(0, _HERE)
sys.path.insert(0, _MY_EXT)

from halite_engine import HaliteEngine

_BOT_DIRS = {
    'rl_v3': os.path.join(_MY_EXT, 'rl_v3'),
    'rl_v2': os.path.join(_MY_EXT, 'rl_v2'),
}


def _bot_cmd(bot_dir: str, model_path: str = None) -> str:
    bot_py  = os.path.join(bot_dir, 'rl_bot.py')
    weights = model_path or os.path.join(bot_dir, 'checkpoints', 'model_final_weights.pt')
    return f'python -u "{bot_py}" --model "{weights}"'


def run_vs_bot(model_path, opponent_bot, model2, width, height, seed, replay_file):
    """rl_v4 vs another RL bot via the hlt protocol (subprocess), saving a replay."""
    bot0 = _bot_cmd(_HERE, model_path)
    bot1 = _bot_cmd(_BOT_DIRS[opponent_bot], model2)

    eng = HaliteEngine(width=width, height=height, num_players=2,
                       seed=seed, verbose=False)
    print(f"Running rl_v4 vs {opponent_bot}  map={width}x{height}  seed={seed}")
    os.makedirs(os.path.dirname(os.path.abspath(replay_file)), exist_ok=True)
    results = eng.run([bot0, bot1], replay_file=replay_file)

    print("\nFinal results:")
    for rank, (pid, halite) in enumerate(results, 1):
        name = 'rl_v4' if pid == 0 else opponent_bot
        print(f"  #{rank} Player {pid} ({name}): {halite:,} halite")
    print(f"  dropoffs: rl_v4={eng._number_dropoffs.get(0, 0)}  "
          f"{opponent_bot}={eng._number_dropoffs.get(1, 0)}")
    print(f"\nReplay saved: {replay_file}")
    return replay_file


def run_vs_greedy(model_path, width, height, seed, replay_file):
    """rl_v4 (player 0) vs the scripted greedy opponent via HaliteEnv."""
    import torch
    from rl_model import ActorCritic
    from rl_env import HaliteEnv
    from rl_features import target_dropoffs, spawn_econ_ok

    model = ActorCritic()
    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'] if isinstance(ckpt, dict)
                          and 'model_state' in ckpt else ckpt)
    model.eval()

    env = HaliteEnv(width=width, height=height, num_players=2, seed=seed,
                    opponent_policy='greedy')
    obs, _ = env.reset()
    eng = env.engine
    print(f"Running rl_v4 vs greedy  map={width}x{height}  seed={seed}  turns={eng.max_turns}")

    done = False
    while not done:
        tgt = target_dropoffs(eng.width, eng.height)
        spawn = spawn_econ_ok(eng.players[0]['energy'], len(obs),
                              eng.max_turns - eng.turn,
                              len(eng.players[0]['dropoffs']), tgt)
        actions = {}
        for ship_id, (spatial, scalars, mask) in obs.items():
            with torch.no_grad():
                a, _, _ = model.select_action(torch.from_numpy(spatial),
                                              torch.from_numpy(scalars), mask=mask)
            actions[ship_id] = a
        obs, _, done, info = env.step(actions, spawn=spawn)

    os.makedirs(os.path.dirname(os.path.abspath(replay_file)), exist_ok=True)
    eng.write_replay(replay_file, player_names={0: 'RL_v4', 1: 'Greedy'})

    results = sorted([(pid, eng.players[pid]['energy']) for pid in eng.players],
                     key=lambda x: -x[1])
    print("\nFinal results:")
    for rank, (pid, halite) in enumerate(results, 1):
        name = 'rl_v4' if pid == 0 else 'greedy'
        print(f"  #{rank} Player {pid} ({name}): {halite:,} halite")
    print(f"  dropoffs: rl_v4={len(eng.players[0]['dropoffs'])}")
    print(f"\nReplay saved: {replay_file}")
    return replay_file


def main():
    parser = argparse.ArgumentParser(description='Generate an rl_v4 Halite III replay')
    parser.add_argument('--model', default=os.path.join(_HERE, 'checkpoints',
                                                        'model_final_weights.pt'),
                        help='rl_v4 weights .pt (player 0)')
    parser.add_argument('--opponent-bot', default='rl_v3', choices=list(_BOT_DIRS),
                        help='Archived RL bot for player 1 (default: rl_v3)')
    parser.add_argument('--model2', default=None,
                        help="Override the opponent bot's weights path")
    parser.add_argument('--greedy', action='store_true',
                        help='Play vs the scripted greedy opponent instead of an RL bot')
    parser.add_argument('--width',  type=int, default=32)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--seed',   type=int, default=None)
    parser.add_argument('--output', default=None, help='Output .hlt path (auto-named if omitted)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    if args.output is None:
        ts  = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tag = 'rl_v4_vs_greedy' if args.greedy else f'rl_v4_vs_{args.opponent_bot}'
        out = os.path.join(_HERE, 'replays', f'replay-{ts}-{seed}-{tag}.hlt')
    else:
        out = args.output

    if args.greedy:
        run_vs_greedy(args.model, args.width, args.height, seed, out)
    else:
        run_vs_bot(args.model, args.opponent_bot, args.model2,
                   args.width, args.height, seed, out)

    print("\nView with:")
    print(f"  python replay_viewer.py \"{out}\"")


if __name__ == '__main__':
    main()
