# rl_v12 — continual league learning (in progress)

*(Builds on [`../rl_v11/README.md`](../rl_v11/README.md) and the shared code in `../rl_v10/`.)*

## Idea
rl_v11 stops learning once training ends: its weights are frozen and `bot.py` only runs inference. rl_v12
keeps learning as competition grows, without the usual risk of online learning (silently getting worse):

1. **Continual training.** PPO keeps running in the background (`rl_v10/train.py --league`), always with
   sampled actions, which is where exploration and new variations come from.
2. **A growing league.** Opponents are listed in `league.json`, which is re-read every iteration, so a new
   opponent can be added while training runs. It currently holds the current champion, rl_v11 (greedy and
   sampled), rl_v10, V71, rl_v9 and rl_v8 … rl_v1, plus mirror self-play (20%) and rl_v12's own past
   snapshots (15%).
3. **Prioritized opponents.** The remaining 65% of games go to listed opponents, weighted by
   `max(0.1, 1 − win rate)^1.5` (prioritized fictitious self-play). Opponents rl_v12 struggles against come
   up more often. Running win rates are kept in `checkpoints/league_stats.json`.
4. **Champion gate** (`gate.py`). Every 20 iterations the newest snapshot (greedy) plays the current champion
   (greedy) for 48 real-engine games. It is promoted to `checkpoints/best.pt` only with at least 29 wins
   (60%). The champion is also a league opponent, so each promotion raises the bar for the learner.
5. **Greedy play.** `best.pt` stores `greedy = True`; learning samples, competing is greedy (see rl_v11's
   greedy vs sampled study).

Everything stays fully learned: the same network, masks and play code as rl_v11, with no hand-coded behavior.

## Start state
* Learner and first champion: rl_v11's weights (`checkpoints/init.pt`, `checkpoints/best.pt`).
* Same network (64 ch × 6 blocks), PPO settings and rewards as rl_v11.

## Running it
From `my_extension/rl_v10/`:
```bash
python train.py --out ../rl_v12/checkpoints --resume ../rl_v12/checkpoints/init.pt --ch 64 --blocks 6 \
  --spawn-warmup 0 --curr-iters 0 --lr 3e-4 --epochs 3 --mb 128 --target-kl 0.05 --ent 0.005 \
  --league ../rl_v12/league.json --iters 100000
```
and from `my_extension/`: `python rl_v12/gate.py`. To resume after a stop, pass
`--resume ../rl_v12/checkpoints/latest.pt` instead of `init.pt`. To add an opponent, append it to
`league.json` (`{"name": ..., "ext": <engine bot>}` or `{"name": ..., "net": <checkpoint>, "greedy": true}`).

Promotions are logged in `checkpoints/gate_log.jsonl`. Play the current champion with
`python rl_v10/bot.py --model rl_v12/checkpoints/best.pt --name rl_v12`.

## Expectations
rl_v11 already beats every existing opponent 93–99%, so gains against today's opponents are likely small.
The value of rl_v12 is that it keeps adapting as new opponents are added, and only ships verified
improvements.

## Results
*(updated as the run progresses)*

### Run 1 (iterations 1–60, lr 3e-4, started from rl_v11)
Champion gate, each candidate (greedy) vs the current champion (greedy), 48 real-engine games:

| Iteration | Wins | Outcome |
|---|---|---|
| 20 | 32/48 | **Promoted** (champion was rl_v11) |
| 40 | 16/48 | Rejected |
| 60 | 17/48 | Rejected |

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
learning rate, so it refines the gain instead of drifting away from it. Iteration numbering continues
(next gates at 80, 100, …).
