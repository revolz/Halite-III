# rl_v11 — the widened network

*Easy reference: **the widened network**: rl_v10's network widened from 32 to 64 channels (2× channels,
3.7× parameters: 143k → 534k), then trained against rl_v10, V71, rl_v9 and itself. Succeeded by rl_v12
(the continual learner), which started from rl_v11.*

*(Predecessor: [`../rl_v10/README.md`](../rl_v10/README.md), whose code rl_v11 reuses.)*

## Goal
A bigger network than rl_v10, trained against more varied opponents including rl_v10 itself, that beats
rl_v10, V71 and rl_v9. Every decision (ship moves, dropoffs, spawning) stays learned, with no hand-coded
behaviour.

## Final verdict (2026-10-20): GOAL MET

`checkpoints/best.pt` (= training snapshot 80) plays **greedy** by default: each ship takes its most likely
action. The setting is stored in the checkpoint (`extra['greedy'] = True`); `bot.py --sample` restores sampled
play. Fresh seeds (seed0 3400000) through the real engine (`halite_engine.py`, all bots as subprocesses),
seats alternated:

| Opponent | Wins | 95% CI (Wilson) | Mean halite rl_v11 / opponent |
|---|---|---|---|
| **rl_v10** | **193/200** | 93.0–98.3% | 47,953 / 30,412 |
| **V71** | **98/100** | 93.0–99.4% | 50,309 / 15,091 |
| **rl_v9** | **99/100** | 94.6–99.8% | 48,994 / 12,784 |

For comparison, rl_v10 scored 79/100 vs V71 and 89/100 vs rl_v9.

```bash
# reproduce (from my_extension/); best.pt plays greedy by default
python rl_v10/evaluate.py --opponent rl_v10 --name rl_v11 --games 200 --seed0 3400000 --model rl_v11/checkpoints/best.pt
python rl_v10/evaluate.py --opponent v71    --name rl_v11 --games 100 --seed0 3400000 --model rl_v11/checkpoints/best.pt
python rl_v10/evaluate.py --opponent rl_v9  --name rl_v11 --games 100 --seed0 3400000 --model rl_v11/checkpoints/best.pt
```

### Greedy vs sampled play
The same network on identical maps (seed0 3400000), with replays to count collisions:

| Matchup | Greedy | Sampled |
|---|---|---|
| vs rl_v10 | 193/200 (96.5%), 47,953 vs 30,412 | 187/200 (93.5%), 44,853 vs 30,983 |
| vs V71 | 98/100, 50,309 vs 15,091 | 99/100, 52,513 vs 21,692 |
| vs rl_v9 | 99/100, 48,994 vs 12,784 | 97/100, 45,856 vs 15,111 |
| Friendly crashes per game (vs rl_v10 / V71 / rl_v9) | 2.48 / 2.30 / 2.66 | 2.90 / 2.99 / 2.83 |

**Head-to-head, greedy vs sampled: greedy won 61/100 (95% CI 51.2–70.0%), 46,561 vs 42,827 mean halite,
32/50 and 29/50 by seat.** Greedy beats sampled directly and is at least as strong against every opponent;
the paired differences against single opponents are not individually significant (sign test p = 0.26, 1.0,
0.63). Training still samples, which is how the policy explores; greedy is only the play-time choice.

The first verification, with sampled play on seed0 2900000, gave 187/200 vs rl_v10 (89.2–96.2%), 98/100 vs
V71 and 97/100 vs rl_v9 (`results/final_eval.csv`).

Per-game rows for the greedy/sampled study: `results/greedy_vs_sampled/`; screening: `results/screen.csv`;
periodic checks: `results/eval_summary.txt` and `results/eval_history.csv`.

## What changed from rl_v10
* **Network:** 64 channels × 6 residual blocks, 534k parameters (rl_v10: 32 × 6, 143k). Same inputs, heads and
  masks (`rl_v10/model.py`, `rl_v10/obs.py`).
* **Start:** `widen.py` copies rl_v10's learned weights into the wide network function-preservingly
  (Net2Net-style): old→old weights copied, weights into new channels small random, every weight from a new
  channel into an old channel or a head zero. The widened net reproduces rl_v10's outputs exactly (max
  difference 0.0), and the new channels receive gradients. No imitation step.
* **Opponents per 16-game batch:** ~25% frozen rl_v10 (in-process), 20% real V71, 15% real rl_v9
  (subprocesses via `rl_v10/ext.py`), the rest mirror self-play and rl_v11's own snapshots.
* **Same PPO and rewards as rl_v10** (see its README); no spawn warm-up or curriculum, since the widened
  network already plays well.

## Is it fully learned?
Yes. It plays through the shared `rl_v10/bot.py`: each turn the board is encoded, the network runs once, and
every ship action plus the spawn decision is taken from its outputs. By default that is the most likely action
(greedy, stored in the checkpoint); with `--sample`, actions are drawn from the network's probabilities. The
only constraints are the game's legality masks and the mask on spawning onto our own ship. There is no endgame
recall, collision resolver or spawn rule. The reward function is a design choice.

Ships coordinate without communicating: one network scores every ship from the whole board in a single pass,
so all ships follow the same learned conventions, and training penalizes friendly crashes (lost cargo and
hull). Measured over 400 replayed games per mode: about 2.3–2.7 open-map friendly crashes per game when greedy and
2.8–3.0 when sampled.

## Training log
* 80 iterations × 16 games = 1,280 training games, ≈5.3 h of compute (≈4 min per iteration, 4 CPU cores).
* Training-game win rates (per 20 iterations):

  | Iterations | vs rl_v10 | vs V71 | vs rl_v9 |
  |---|---|---|---|
  | 1–20 | 0.67 | 0.83 | 0.92 |
  | 21–40 | 0.84 | 0.92 | 0.99 |
  | 41–60 | 0.92 | 0.98 | 0.98 |
  | 61–80 | 0.91 | 0.90 | 1.00 |

* Real-engine checks (24 games vs rl_v10, 8 vs each of the others):

  | Iteration | vs rl_v10 | vs V71 | vs rl_v9 |
  |---|---|---|---|
  | 20 | 20/24 | 7/8 | 8/8 |
  | 40 | 20/24 | 8/8 | 8/8 |
  | 60 | 23/24 | 7/8 | 8/8 |
  | 80 | 23/24 | 8/8 | 8/8 |

* Training was stopped at iteration 80 because the curve had flattened. Screening on 48 fresh games vs rl_v10:
  snapshots 60 / 70 / 80 scored 44 / 45 / 45; snapshot 80 was kept and verified above.

## Caveats
* rl_v11 trained directly against all three opponents, so part of its edge may be opponent-specific.
* Like rl_v10, it learned under this engine's rules, including its quirks (double-counted dropoff credit,
  bots told the game lasts 500 turns).

## Files and commands
`widen.py` builds the starting checkpoint; `eval_loop.sh` runs the periodic checks. Training (from
`my_extension/rl_v10/`):
```bash
python ../rl_v11/widen.py
python train.py --out ../rl_v11/checkpoints --resume ../rl_v11/checkpoints/init.pt --ch 64 --blocks 6 \
  --spawn-warmup 0 --curr-iters 0 --lr 3e-4 --epochs 3 --mb 128 --target-kl 0.05 --ent 0.005 \
  --fixed-opp "$(pwd)/checkpoints/best.pt" --p-fixed 0.25 --p-v71 0.2 --p-v9 0.15
```
Play: `python rl_v10/bot.py --model rl_v11/checkpoints/best.pt --name rl_v11` (greedy by default; add `--sample` for sampled play).
