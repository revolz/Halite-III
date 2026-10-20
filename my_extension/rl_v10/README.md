# rl_v10 — tabula-rasa self-play RL that beats V71 and rl_v9

*(For the full rl_v1→v9 lineage see [`../README.md`](../README.md).)*

## Goal
Beat both **V71** (`Year 2019/MyBot - V71`, the strongest 2019 hand-coded bot) and **rl_v9** (the previous
champion, 57% vs V71) in 1v1 on 32x32 maps, with a bot whose every decision — ship moves, dropoff
construction and spawning — is learned. No imitation of any other bot and no hand-coded behaviour.

## Final verdict (2026-10-19): GOAL MET

`checkpoints/best.pt` (= training snapshot 280), 100 fresh-seed games per opponent through the real engine
(`halite_engine.py`, every bot a subprocess), seats alternated, sampled actions:

| Opponent | Wins | 95% CI (Wilson) | Mean halite rl_v10 / opponent | As seat 0 / seat 1 |
|---|---|---|---|---|
| **V71** | **79/100** | 70.0–85.8% | 38,529 / 25,889 | 40/50, 39/50 |
| **rl_v9** | **89/100** | 81.4–93.7% | 38,518 / 12,854 | 45/50, 44/50 |

Both lower bounds clear 50%. Reproduce (from `my_extension/`):

```bash
python rl_v10/evaluate.py --opponent v71   --games 100 --seed0 900000 --model rl_v10/checkpoints/best.pt
python rl_v10/evaluate.py --opponent rl_v9 --games 100 --seed0 900000 --model rl_v10/checkpoints/best.pt
```

Per-game rows: `results/final_eval.csv`; candidate screening: `results/screen.csv`; the 8-game
checkpoint evals during training: `results/eval_summary.txt`.

## What is learned and what is not
* **Learned:** every ship action (stay / N / E / S / W / build dropoff) and the spawn decision, sampled from
  the network each turn (`bot.py`).
* **Masks (not strategy):** the game's own legality — a ship that can't pay the move cost can't move, a
  dropoff must be affordable, spawning needs 1000 halite — plus one dominated action: spawning onto our own
  ship (destroys both ships, gains nothing).
* **Training-only scaffolding**, absent at play time: a spawn-exploration warm-up and a starting-bank
  curriculum (see below).
* **Reward shaping** is designed (below); the ship shaping is potential-based, so it cannot change the
  optimal policy.

## Architecture (`model.py`, 143k parameters)
* **Input** (`obs.py`): the whole 32x32 board as 20 planes in the seat's canonical frame (boards are
  mirrored so the bot always plays from the left): halite, move cost, own/enemy ships and cargo, own/enemy
  structures, own shipyard, distance to nearest own/enemy structure, enemy/own ships within inspiration
  radius 4, ships that can't afford to move, and constant planes for turns left, both banks, both ship
  counts and remaining halite.
* **Trunk:** 3x3 conv → 6 residual blocks of 32 channels with circular (torus) padding; each block adds a
  global-context path (mean+max pool → linear → broadcast), so every cell sees the whole board.
* **Heads:** ship policy = 6 logits per cell (read where we have ships, all ships in one pass); ship value =
  1 per cell; spawn = 2 logits + value from the shipyard cell plus global pooling.

## Training (`train.py`)
* **Simulator** (`sim.py`): a batched numpy re-implementation of `halite_engine.py` including its quirks,
  16 games in lock-step. `verify_sim.py` replays real engine games and asserts identical ships, cargo,
  inspiration, banks and every halite cell on every turn (verified on V71/rl_v9 and random-bot games).
* **Opponents:** each 16-game batch mixes mirror self-play, frozen past snapshots, and the **real V71 and
  rl_v9 bots** (25% each) run as subprocesses through `ext.py`, which speaks the engine's exact protocol and
  constants (an rl_v9-vs-rl_v9 game reproduces the real engine's final scores exactly).
* **PPO** with per-ship GAE (each ship's own trajectory, γ=0.995, λ=0.95), clip 0.2, lr 5e-4, 3 epochs,
  minibatch 128 boards, entropy 0.005, KL stop 0.05; bf16 autocast on the CPU's AMX units.
* **Rewards** (units of 1000 halite):

  | Decision | Reward |
  |---|---|
  | Ship step | banked + γ·Φ' − Φ, Φ = cargo × (0.25 + 0.65·max(0, 1 − dist_home/16)) |
  | Ship wrecked | − Φ − own hull + 0.5 × enemy cargo destroyed + enemy hulls destroyed (hull = 1000·min(1, turns_left/200)) |
  | Build dropoff | bank delta + 0.25 × halite later deposited there |
  | Spawn | lifetime halite banked by the new ship − 1000 |

* **Compute:** 4 CPU cores, no GPU; 280 iterations × 16 games = 4,480 training games.

## What made it stronger (run log)
Every stall was diagnosed from logs or replays and its cause removed:

| Iteration | Problem | Fix |
|---|---|---|
| 1–7 | Learned never to spawn (untrained ships don't repay 1000) | Spawn-exploration warm-up; spawn head untrained meanwhile |
| ~18 | New ships killed by spawning onto them | Masked the dominated self-spawn |
| ~20 | Too few ships to learn from | Starting-bank curriculum (up to 30k, annealed to 5000 by iteration 150) |
| ~36 | Loaded ships didn't head home | Distance-aware potential shaping |
| ~36 | Empty ships rammed for free hull credit | Charge own hull symmetrically |
| ~46 | Updates too timid (KL ≈ 0.005) | Higher lr, more epochs, smaller minibatches |
| 106 | Ships competent (~20k deposited/game) | Spawn learning on (head reset, hoarding snapshots retired) |
| 115 | Not learning to handle the actual opponents | Real V71/rl_v9 in the training mix |
| ~170 | Losing fleets scuttled before turn 400 to dodge a per-ship outcome penalty | Removed that term |

Real-engine checkpoints (8 games each, from `results/eval_summary.txt`):

| Iteration | vs V71 | vs rl_v9 |
|---|---|---|
| 120 | 0/8, 8.3k vs 36.9k | 0/8, 7.4k vs 30.6k |
| 160 | 0/7, 14.4k vs 46.7k | 1/8, 14.2k vs 30.6k |
| 200 | 2/8, 23.0k vs 34.5k | 4/8, 19.4k vs 22.8k |
| 240 | 4/8, 40.6k vs 38.7k | 8/8, 39.2k vs 10.3k |
| 280 | 7/8, 26.5k vs 17.1k | 8/8, 24.8k vs 11.7k |

Screening on 24 fresh games vs V71: snapshots 260 / 270 / 280 scored 14 / 20 / 19 wins; 280 was kept
(44 wins in 56 real games across all checks), then verified on the 100+100 games above.

## Caveats
* rl_v10 trained against V71 and rl_v9 directly, so part of its edge may exploit their specific
  weaknesses; strength against other bots is not established.
* The engine counts a dropoff's credit twice and tells bots the game lasts 500 turns (it lasts 400); the
  simulator copies both, so rl_v10 learned under those rules.
* The dropoff credit (0.25) and kill/hull pricing are shaping choices that can bias strategy.

## Files
`sim.py` simulator · `verify_sim.py` engine-equivalence check · `obs.py` encoder · `model.py` network ·
`train.py` PPO trainer · `ext.py` real-bot opponents · `bot.py` live bot · `evaluate.py` real-engine
matches · `eval_loop.sh` periodic evals · `diag.py` simulator diagnostics.

## Shared tooling (also used by rl_v11 and rl_v12)
rl_v11 and rl_v12 reuse this folder's code with their own checkpoints. Options added after rl_v10 shipped
(rl_v10's own behaviour is unchanged):

| Script | Option | Meaning |
|---|---|---|
| `bot.py` | `--greedy` / `--sample` | Most-likely action vs sampled action. Default: the checkpoint's `extra['greedy']` (rl_v10: sampled; rl_v11/rl_v12: greedy) |
| `bot.py` | `--name NAME` | Name announced to the engine (shown in replays) |
| `evaluate.py` | `--opponent v71\|rl_v9\|rl_v10` | Built-in opponents |
| `evaluate.py` | `--opponent LABEL --opp-model PATH [--opp-greedy\|--opp-sample]` | Any checkpoint as the opponent, e.g. rl_v11 greedy vs rl_v11 sampled |
| `evaluate.py` | `--name`, `--greedy`/`--sample`, `--replays DIR`, `--csv FILE` | Label and play mode of the evaluated bot; save replays / per-game rows |
| `train.py` | `--fixed-opp PATH --p-fixed F` | Share F of games vs a frozen in-process network |
| `train.py` | `--p-v71 F`, `--p-v9 F` | Share of games vs the real V71 / rl_v9 bots |
| `train.py` | `--league FILE [--league-ema A]` | League training (rl_v12): opponent list re-read every iteration, prioritized by win rate |
| `train.py` | `--reset-spawn-head` | Zero the spawn head's last layer when resuming |
| `ext.py` | opponents `v71`, `rl_v9`, `rl_v1`…`rl_v8` (not v6) | Engine-protocol bots usable as simulator opponents |

```bash
# rl_v11 greedy vs rl_v11 sampled, 100 games (from my_extension/)
python rl_v10/evaluate.py --model rl_v11/checkpoints/best.pt --name rl_v11_greedy --greedy \
  --opponent rl_v11_sampled --opp-model rl_v11/checkpoints/best.pt --opp-sample --games 100
```

```bash
# train (from rl_v10/), as in the final run
python train.py --ch 32 --blocks 6 --envs 16 --lr 5e-4 --epochs 3 --mb 128 --target-kl 0.05 \
    --ent 0.005 --spawn-warmup 106 --p-v71 0.25 --p-v9 0.25
# play
python rl_v10/bot.py --model rl_v10/checkpoints/best.pt
```
The final run reached its result through resumes with the settings changed as in the run log; only
`checkpoints/best.pt` is committed (other checkpoints stay local).
