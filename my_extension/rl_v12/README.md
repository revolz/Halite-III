# rl_v12 — the continual learner (strongest bot)

*Easy reference: **the continual learner**: rl_v11 kept training in a league of every earlier bot; a new
version replaces the champion only after provably beating it. The strongest bot in the repo.*

*(Builds on [`../rl_v11/README.md`](../rl_v11/README.md) and the shared code in `../rl_v10/`.)*

## Final verdict (2026-10-21): stronger than rl_v11

`checkpoints/best.pt` = training snapshot 160, the last of three verified promotions in 300 iterations of
continual training. Fresh seeds (seed0 7700000), real engine (`halite_engine.py`, all bots as subprocesses),
seats alternated, both sides greedy:

| Opponent | Wins | 95% CI (Wilson) | Mean halite rl_v12 / opponent |
|---|---|---|---|
| **rl_v11** | **157/200 (78.5%)** | 72.3–83.6% | 50,083 / 43,184 |
| **Previous champion (snapshot 80)** | **131/200 (65.5%)** | 58.7–71.7% | 48,910 / 45,608 |
| **V71** | **100/100** | 96.3–100% | 55,451 / 10,180 |
| **rl_v9** | **98/100** | 93.0–99.4% | 52,464 / 12,742 |

**Final round robin of the top five** (20 games per pair, `../README.md`): rl_v12 went **78-2**, sweeping
rl_v10, rl_v9 and V71 20-0 and beating rl_v11 18-2.

Promotion chain: rl_v11 → snapshot 20 (69.0% vs rl_v11) → snapshot 80 (65.0% vs snapshot 20, 73.5% vs
rl_v11) → snapshot 160 (65.0% vs snapshot 80, 78.5% vs rl_v11). No candidate passed the gate in the 140
iterations after snapshot 160 (180–300), so the run was stopped at iteration 300 as planned.

```bash
# reproduce (from my_extension/); best.pt plays greedy by default
python rl_v10/evaluate.py --model rl_v12/checkpoints/best.pt --name rl_v12 --opponent rl_v11 \
  --opp-model rl_v11/checkpoints/best.pt --opp-greedy --games 200 --seed0 7700000
python rl_v10/evaluate.py --model rl_v12/checkpoints/best.pt --name rl_v12 --opponent v71   --games 100 --seed0 7700000
python rl_v10/evaluate.py --model rl_v12/checkpoints/best.pt --name rl_v12 --opponent rl_v9 --games 100 --seed0 7700000
```
Per-game rows are in `results/` (`c160_vs_*.csv` for the final champion, `c80_vs_*.csv`, `snap20_vs_v11.csv`,
`v12champ_*.csv` for earlier champions, `gate_log.jsonl` for every gate decision).

### What changed in its play
From replays on one random map (seed 78231680), rl_v12 runs a bigger economy than rl_v11:

| | rl_v12 (snapshot 160) | rl_v11 (earlier replays) |
|---|---|---|
| Ships built per game | 43–49 | 34–35 |
| Dropoffs per game | 2–5 | usually 0 |
| Halite mined per game | 94k–123k | 82k–104k |

Against V71 it mined about twice as much as V71 in both games and held it to 738 and 3,300 halite. This
comparison is from a handful of replays, so treat it as an illustration rather than a measured average.

## Idea
rl_v11 stops learning once training ends: its weights are frozen and `bot.py` only runs inference. rl_v12
keeps learning as competition grows, without the usual risk of online learning (silently getting worse):

1. **Continual training.** PPO keeps running in the background (`rl_v10/train.py --league`), always with
   sampled actions, which is where exploration and new variations come from.
2. **A growing league.** Opponents are listed in `league.json`, which is re-read every iteration, so a new
   opponent can be added while training runs. It holds the current champion, rl_v11 (greedy and sampled),
   rl_v10, V71, rl_v9 and rl_v8 … rl_v1 (not v6).
3. **Opponent mix.** Final settings: mirror self-play 15%, rl_v12's own past snapshots 10%, the champion a
   fixed 30% (`"focus"` in `league.json`), and the remaining 45% across the listed opponents weighted by
   `max(0.1, 1 − win rate)^1.5` (prioritized fictitious self-play), so opponents rl_v12 struggles against come
   up more often. Running win rates are kept in `checkpoints/league_stats.json`. (Runs 1–2 used 20% self-play,
   15% snapshots and no champion focus.)
4. **Champion gate** (`gate.py`). Every 20 iterations the newest snapshot (greedy) plays the current champion
   (greedy) for 48 real-engine games; with at least 29 wins it plays 152 more, and it is promoted to
   `checkpoints/best.pt` only with at least 115/200 overall. After a rejection the learner restarts from the
   champion. The champion is also a league opponent, so each promotion raises the bar for the learner.
5. **Greedy play.** `best.pt` stores `greedy = True`; learning samples, competing is greedy (see rl_v11's
   greedy vs sampled study).

Everything stays fully learned: the same network, masks and play code as rl_v11, with no hand-coded behavior.

## Start state
* Learner and first champion: rl_v11's weights (`checkpoints/init.pt`, `checkpoints/best.pt`).
* Same network (64 ch × 6 blocks), PPO settings and rewards as rl_v11.

## Running it
Play the champion: `python rl_v10/bot.py --model rl_v12/checkpoints/best.pt --name rl_v12` (from
`my_extension/`; greedy by default, `--sample` for sampled play).

To continue training (only `best.pt` is committed, so start a new learner from it), from `my_extension/rl_v10/`:
```bash
python -c "import sys; sys.path.insert(0, '.'); from model import load, save; n, _ = load('../rl_v12/checkpoints/best.pt'); save(n, '../rl_v12/checkpoints/init.pt', {'iter': 0})"
python train.py --out ../rl_v12/checkpoints --resume ../rl_v12/checkpoints/init.pt --ch 64 --blocks 6 \
  --spawn-warmup 0 --curr-iters 0 --lr 1e-4 --epochs 3 --mb 128 --target-kl 0.05 --ent 0.005 \
  --league ../rl_v12/league.json --reset-file ../rl_v12/checkpoints/reset_to --iters 100000
```
and, alongside it from `my_extension/`, `python rl_v12/gate.py` (screen 48 games at 29 wins, confirm to
115/200; writes `checkpoints/reset_to` after a rejection). To resume an interrupted run, pass
`--resume ../rl_v12/checkpoints/latest.pt` instead. To add an opponent, append it to `league.json`
(`{"name": ..., "ext": <engine bot>}` or `{"name": ..., "net": <checkpoint>, "greedy": true}`). Every gate
decision is logged in `checkpoints/gate_log.jsonl`.

## Lessons
* **Continual learning works only with a gate.** Training drifted after every gain; 12 of the 15 gated
  snapshots were not better than their champion. Without the gate the run would have ended weaker than it
  started.
* **Screen, then confirm.** A 48-game screen let one lucky snapshot through (120); two equal bots reach
  29/48 about 10% of the time. Promotion now needs 115/200.
* **Restart from the champion.** The second and third gains came only after restarting from the champion at
  a lower learning rate (1e-4) and resetting the learner after each failed gate.
* **Diminishing returns.** Promotions came 20, 60 and 80 iterations apart, then none in 140 iterations,
  although each one was still a solid step (65–69% head-to-head).
* **Not rock-paper-scissors.** Each champion also beat rl_v11 by more than the last (69.0% → 73.5% →
  78.5%), so the gains are general rather than counters to the previous champion.

## Run log
Total: 300 iterations × 16 games = 4,800 training games, ≈23.5 h of compute on 4 CPU cores, plus the gate
and verification games.

### Run 1 (iterations 1–60, lr 3e-4, started from rl_v11)
Champion gate, each candidate (greedy) vs the current champion (greedy), 48 real-engine games:

| Iteration | Wins | Outcome |
|---|---|---|
| 20 | 32/48 (66.7%) | **Promoted** (champion was rl_v11) |
| 40 | 16/48 (33.3%) | Rejected |
| 60 | 17/48 (35.4%) | Rejected |

Larger checks of the champion (snapshot 20), fresh seeds, real engine, greedy:

| Opponent | Wins | 95% CI | Mean halite rl_v12 / opponent |
|---|---|---|---|
| **rl_v11 (greedy)** | **138/200** | 62.3–75.0% | 46,756 / 44,275 (69/100 from each seat) |
| **V71** | **100/100** | 96.3–100% | 54,054 / 15,516 |
| **rl_v9** | **100/100** | 96.3–100% | 48,197 / 11,615 |

So the first promotion was real: continual training produced a bot clearly stronger than rl_v11 within 20
iterations. Training then drifted. In training games the sampled learner won only ~40–55% against the greedy
champion and rl_v11 while beating everyone else ~100%, and snapshots 40 and 60 lost to the champion. The gate
did its job and kept the better bot.

### Run 2 (from iteration 60): restart from the champion at lr 1e-4
After two failed gates in a row, the learner was restarted from the champion's weights with a 3× lower
learning rate, so it refines the gain instead of drifting away from it. Iteration numbering continues.

| Iteration | Wins vs champion | Outcome |
|---|---|---|
| 80 | 30/48 (62.5%) | **Promoted** (champion was snapshot 20) |
| 100 | 18/48 (37.5%) | Rejected |

Larger checks of champion snapshot 80, fresh seeds (seed0 7300000), real engine, greedy:

| Opponent | Wins | 95% CI | Mean halite rl_v12 / opponent |
|---|---|---|---|
| **Previous champion (snapshot 20)** | **130/200** | 58.2–71.3% | 49,524 / 43,994 |
| **rl_v11 (greedy)** | **147/200** | 67.0–79.1% | 50,299 / 44,158 |
| **V71** | **98/100** | 93.0–99.4% | 56,514 / 17,424 |
| **rl_v9** | **99/100** | 94.6–99.8% | 52,541 / 11,845 |

Each promotion is a real gain: rl_v11 → snapshot 20 (69% head-to-head) → snapshot 80 (65% vs snapshot 20,
73.5% vs rl_v11). The same pattern also showed: a gain within ~20 iterations of starting from a champion,
then drift.

### Run 3 (from iteration 100): champion focus + automatic reset
* `league.json` gives the champion a fixed 30% of games (self-play 15%, snapshots 10%, the remaining 45%
  prioritized across the whole pool), so training concentrates on the matchup the gate measures.
* After every rejected gate, `gate.py` writes `checkpoints/reset_to` and the trainer (`--reset-file`)
  restarts from the champion's weights with a fresh optimizer, so drift never accumulates past one gate
  interval.
* Learning rate 1e-4, starting from champion snapshot 80.

| Iteration | Wins vs champion | Outcome |
|---|---|---|
| 120 | 29/48 (60.4%) | Promoted at exactly the threshold, then **demoted**: a 200-game confirmation gave 105/200 vs snapshot 80 (52.5%, 95% CI 45.6–59.3%), i.e. no better. Snapshot 80 restored as champion |

Two equal bots reach 29/48 about 10% of the time, so the 48-game gate alone let a lucky snapshot through. Since
then the gate has two stages: a candidate that passes the 48-game screen plays 152 more games and is
promoted only with at least 115/200 overall (57.5%, where the 95% CI lower bound is above 50%).

| Iteration | Screen | Confirmation | Total | Outcome |
|---|---|---|---|---|
| 140 | 20/48 (41.7%) | — | — | Rejected; learner reset to the champion |
| 160 | 32/48 (66.7%) | 98/152 (64.5%) | **130/200 (65.0%, 95% CI 58.2–71.3%)** | **Promoted** (champion was snapshot 80) |
| 180 | 15/48 (31.3%) | — | — | Rejected; learner reset to the champion |
| 200 | 18/48 (37.5%) | — | — | Rejected; learner reset to the champion |
| 220 | 15/48 (31.3%) | — | — | Rejected; learner reset to the champion |
| 240 | 26/48 (54.2%) | — | — | Rejected (below the 29-win screen); learner reset to the champion |
| 260 | 16/48 (33.3%) | — | — | Rejected; learner reset to the champion |
| 280 | 22/48 (45.8%) | — | — | Rejected; learner reset to the champion |
| 300 | 24/48 (50.0%) | — | — | Rejected; run stopped (end of the planned window) |

Larger checks of champion snapshot 160, fresh seeds (seed0 7700000), real engine, greedy:

| Opponent | Wins | 95% CI | Mean halite rl_v12 / opponent | Snapshot 80 for comparison |
|---|---|---|---|---|
| **rl_v11 (greedy)** | **157/200 (78.5%)** | 72.3–83.6% | 50,083 / 43,184 | 147/200 (73.5%) |
| **Snapshot 80** | **131/200 (65.5%)** | 58.7–71.7% | 48,910 / 45,608 | — |
| **V71** | **100/100** | 96.3–100% | 55,451 / 10,180 | 98/100 |
| **rl_v9** | **98/100** | 93.0–99.4% | 52,464 / 12,742 | 99/100 |

The gain over snapshot 80 replicates on new seeds (65.5% vs the gate's 65.0%) and carries over to rl_v11,
so the progress is general rather than a counter to the previous champion only.
