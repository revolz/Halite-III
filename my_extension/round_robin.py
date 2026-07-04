#!/usr/bin/env python3
"""Round-robin tournament driver for rl_v* bots vs V71 (Year 2019).

Plays every pair 10 times (seeds 1..10) on a 32x32 map, 1v1, using the
pure-Python halite_engine.py. Writes each result immediately to a CSV so
progress can be monitored while it runs.
"""
import csv
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halite_engine import HaliteEngine

HERE = os.path.dirname(os.path.abspath(__file__))

BOTS = {
    "rl_v1": f'python -u "{HERE}/rl_v1/rl_bot.py" --model "{HERE}/rl_v1/checkpoints/best.pt" --deterministic',
    "rl_v2": f'python -u "{HERE}/rl_v2/rl_bot.py" --model "{HERE}/rl_v2/checkpoints/best.pt" --deterministic',
    "rl_v3": f'python -u "{HERE}/rl_v3/rl_bot.py" --model "{HERE}/rl_v3/checkpoints/best.pt" --deterministic',
    "rl_v4": f'python -u "{HERE}/rl_v4/rl_bot.py" --model "{HERE}/rl_v4/checkpoints/best.pt" --deterministic',
    "rl_v5": f'python -u "{HERE}/rl_v5/rl_bot.py" --model "{HERE}/rl_v5/checkpoints/best.pt" --deterministic',
    "rl_v7": f'python -u "{HERE}/rl_v7/rl_bot.py" --model "{HERE}/rl_v7/checkpoints/best.pt" --deterministic',
    "rl_v8": f'python -u "{HERE}/rl_v8/rl_bot.py" --model "{HERE}/rl_v8/checkpoints/best.pt" --deterministic',
    "rl_v9": f'python -u "{HERE}/rl_v9/rl_bot.py" --model "{HERE}/rl_v9/checkpoints/best.pt" --deterministic',
    "V71":   f'python -u "{HERE}/Year 2019/MyBot - V71/MyBot.py"',
}

NUM_MATCHES_PER_PAIR = 10
WIDTH = HEIGHT = 32
OUT_CSV = os.path.join(HERE, "round_robin_results.csv")


def main():
    pairs = list(itertools.combinations(sorted(BOTS.keys()), 2))
    total_games = len(pairs) * NUM_MATCHES_PER_PAIR
    print(f"{len(BOTS)} bots, {len(pairs)} pairs, {NUM_MATCHES_PER_PAIR} games/pair "
          f"= {total_games} games total")

    write_header = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["pair_idx", "game_idx", "seed", "bot_a", "bot_b",
                          "winner", "halite_a", "halite_b", "turns", "seconds", "error"])
        f.flush()

    game_num = 0
    t_start = time.time()
    for pair_idx, (name_a, name_b) in enumerate(pairs):
        for game_idx in range(NUM_MATCHES_PER_PAIR):
            game_num += 1
            seed = game_idx + 1
            t0 = time.time()
            error = ""
            winner = ""
            halite_a = halite_b = ""
            turns = ""
            try:
                engine = HaliteEngine(width=WIDTH, height=HEIGHT, num_players=2,
                                       seed=seed, verbose=False)
                results = engine.run([BOTS[name_a], BOTS[name_b]], replay_file=None)
                by_pid = {pid: h for pid, h in results}
                halite_a = by_pid.get(0, 0)
                halite_b = by_pid.get(1, 0)
                winner = name_a if halite_a > halite_b else (name_b if halite_b > halite_a else "TIE")
                turns = engine.turn
            except Exception as e:
                error = repr(e)
                print(f"  ERROR in {name_a} vs {name_b} seed={seed}: {e}")
            dt = time.time() - t0
            writer.writerow([pair_idx, game_idx, seed, name_a, name_b,
                              winner, halite_a, halite_b, turns, f"{dt:.1f}", error])
            f.flush()
            elapsed = time.time() - t_start
            print(f"[{game_num}/{total_games}] {name_a} vs {name_b} seed={seed} "
                  f"-> winner={winner} ({halite_a} vs {halite_b}) "
                  f"[{dt:.1f}s, elapsed {elapsed/60:.1f}m]")

    f.close()
    print("Done.")


if __name__ == "__main__":
    main()
