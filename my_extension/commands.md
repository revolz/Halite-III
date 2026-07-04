# Halite III – Command Reference

All commands are run from the relevant bot folder unless stated otherwise.

**Shared infrastructure** (run from `my_extension/`):
```bash
cd "C:\Temp\Halite-III\my_extension"
```

**rl_v1 / rl_v2 / rl_v3 bots** (archived – benchmark reference only, do not modify):
```bash
cd "C:\Temp\Halite-III\my_extension\rl_v3"
```

**rl_v4 bot** (active – use this for new training). See `rl_v4/README.md` for
details (9 actions incl. learned dropoff, production-aligned reward, train vs rl_v3):
```bash
cd "C:\Temp\Halite-III\my_extension\rl_v4"
python rl_train.py --opponent rl_v3      # train directly against the rl_v3 bot
```

> The command examples below use `rl_v2` paths; for the active bot substitute
> `rl_v4/` (same CLI). Recent changes are summarized in `my_extension/CHANGELOG.md`.

---

## 1. Run a game

Run from `my_extension/`:

```bash
# Default: starter-kit bot vs itself on a 32×32 map
python run_game.py

# Custom map size
python run_game.py --width 40 --height 40

# Reproducible game (fixed seed)
python run_game.py --seed 12345

# Verbose: print scores and ship counts every 50 turns
python run_game.py --verbose

# Save replay (auto-named, saved to my_extension/replays/)
python run_game.py --replay

# Save replay to a specific file
python run_game.py --replay-file my_game.hlt

# 4-player game (repeat --bot for each player)
python run_game.py --players 4

# Combine: 40×40 map, seed, verbose, auto replay
python run_game.py --width 40 --height 40 --seed 42 --verbose --replay
```

### Run your own RL bot

```bash
# rl_v2 bot vs the starter-kit bot (from my_extension/)
python run_game.py \
  --bot "python rl_v2/rl_bot.py --model rl_v2/checkpoints/best.pt" \
  --bot "python ..\starter_kits\Python3\MyBot.py" \
  --replay --verbose
```

---

## 2. Watch a replay

Run from `my_extension/`:

```bash
# Open the replay viewer (GUI window)
python replay_viewer.py replays\<filename>.hlt

# Example
python replay_viewer.py replays\replay-20260531-162050-42-40-40.hlt
```

**Viewer controls:**
| Button / Key | Action |
|---|---|
| ▶ Play / ⏸ Pause | Toggle auto-play |
| ◀◀ / ▶▶ | Previous / next turn |
| ⏮ / ⏭ | Jump to start / end |
| Speed slider | 0.3× – 12× playback speed |

---

## 3. Train the RL bot

All training commands run from the bot folder (e.g. `my_extension/rl_v2/`).

### Recommended 3-phase training schedule

Train in phases — start easy, then increase difficulty:

**Phase 1 — Learn to mine (idle opponent, 1000 episodes)**
```bash
python rl_train.py --episodes 1000 --checkpoint-dir checkpoints --opponent-policy idle
```
Expected: reward goes from ~0 to positive as bot learns to collect and deposit halite.

**Phase 2 — Add competition (greedy opponent, resume for 1000 more)**
```bash
python rl_train.py --resume checkpoints\model_ep1000.pt --episodes 1000 --checkpoint-dir checkpoints --opponent-policy greedy
```
Expected: reward may dip briefly then recover as bot adapts to an opponent.

**Phase 3 — Fine-tune (optional)**
```bash
python rl_train.py --resume checkpoints\model_ep2000.pt --episodes 1000 --checkpoint-dir checkpoints --opponent-policy greedy
```

> **Note:** `--start-episode` is optional. When omitted, the episode counter is
> auto-detected from the checkpoint so you don't accidentally overwrite history.

### Start training from scratch (single command, idle opponent)

```bash
python rl_train.py --episodes 2000 --checkpoint-dir checkpoints
```

**What happens:**
- Prints progress every 10 episodes: `Episode   10 | reward    0.0 | avg100    0.0 | 12s`
- Saves a checkpoint every 50 episodes: `checkpoints\model_ep50.pt`
- Saves plain weights alongside: `checkpoints\model_ep50_weights.pt`
- Writes a CSV log: `checkpoints\training_log.csv`
- Saves final model when done: `checkpoints\model_final.pt` + `best.pt`

### Resume training from a checkpoint

```bash
# Continue from the last checkpoint (episode counter auto-detected)
python rl_train.py --resume checkpoints\model_ep1000.pt --episodes 1000 --checkpoint-dir checkpoints
```

- `--resume` loads weights **and** optimizer state (training momentum is preserved)
- Episode counter is read automatically from the checkpoint — no manual counting needed
- The CSV log is **appended**, so the full history stays in one file

### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 2000 | Number of episodes to train |
| `--checkpoint-dir` | `checkpoints` | Folder for checkpoints and log |
| `--checkpoint-interval` | 50 | Save a checkpoint every N episodes |
| `--resume` | *(none)* | Path to `.pt` file to resume from |
| `--start-episode` | *(auto)* | Override episode counter start (usually leave unset) |
| `--opponent-policy` | `idle` | `idle` (no moves), `greedy` (scripted heuristic), `random` |
| `--collision-scale` | 20.0 | Fixed penalty added per destroyed p0 ship (on top of cargo lost) |
| `--ent-coef` | 0.25 | Entropy bonus weight — higher = more exploration, prevents policy collapse |
| `--ent-floor` | 0.5 | Entropy floor (nats) — below this threshold an extra penalty kicks in |
| `--ent-floor-coef` | 0.5 | Extra entropy penalty multiplier when below `--ent-floor` |
| `--width` / `--height` | 32 | Map dimensions |
| `--lr` | 3e-4 | Learning rate |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--seed` | *(random)* | Fixed seed for reproducible maps |

---

## 4. Check training progress

### Console output (live)
The training script prints every 10 episodes:
```
Episode    10 | reward      0.0 | avg100      0.0 | deposited      0 | entropy 2.079 | s= 0% n= 0% e= 0% s= 0% w= 0% r= 0% h= 0% p= 0% | 12s
Episode    20 | reward   1500.0 | avg100    800.0 | deposited   1450 | entropy 1.603 | s=18% n=16% e=15% s=10% w=11% r= 8% h=12% p=10% | 24s
```

Columns in order: episode · episode reward · 100-ep rolling average · halite banked · mean entropy · action distribution (stay/north/east/south/west/random/home/**prospect** %) · elapsed seconds.

### CSV log (any time, including after training)
Open `checkpoints\training_log.csv` in Excel, or tail it in a second terminal:
```powershell
# PowerShell: watch the last 5 lines of the log live
while ($true) { Get-Content checkpoints\training_log.csv | Select-Object -Last 5; Start-Sleep 10 }
```

Columns: `episode`, `reward`, `avg100_reward`, `deposited`, `mean_entropy`, `elapsed_sec`, `stay`, `north`, `east`, `south`, `west`, `random`, `home`, `prospect`

- **`deposited`** — raw halite actually banked this episode (ignores reward formula). This is the clearest measure of real performance.
- **`reward`** — shaped reward: `Σ(cargo_after − cargo_before)` for surviving ships + halite deposited − `(collision_scale + cargo_lost)` per destroyed p0 ship.
- **`stay`…`prospect`** — fraction of all actions taken that were each action type this episode (0.0–1.0). Useful for diagnosing policy collapse (e.g. one action near 1.0).

### How to read the entropy column

With 8 actions, max entropy = ln(8) ≈ 2.079 nats.

| Entropy value | What it means | Action |
|---|---|---|
| ~2.08 nats | Perfectly uniform — all 8 actions equally likely | Normal at start |
| 1.0–1.8 nats | Healthy exploration — policy has preferences but still tries things | Good |
| 0.5–1.0 nats | Moderate convergence — learning is working | Good |
| < 0.5 nats | Warning: policy converging hard on a few actions | Watch closely |
| 0.000 nats | **Total collapse** — always picks same action (seen in replay as "always east") | Retrain with higher `--ent-coef` |

### Action guide (rl_v2)

| Action | Index | Type | Behaviour |
|--------|-------|------|-----------|
| stay | 0 | primitive | Stay in place and mine |
| north/east/south/west | 1–4 | primitive | Move one cell |
| random | 5 | meta | Environment picks a random primitive (0–4) |
| home | 6 | meta | Move one step toward nearest deposit (factory/dropoff) |
| **prospect** | **7** | **meta** | **Move one step toward the richest halite cell in the 11×11 window; stay and mine when already there** |

`prospect` is useful when local halite is depleted and ships are crowding the shipyard.
The NN learns *when* to prospect by observing scalars 13–16 (direction + value + distance to the richest nearby cell).

---

## 5. Evaluate a trained checkpoint

Measure how good a checkpoint is: runs N complete games vs a scripted opponent and
reports win rate, mean halite, and halite-per-turn.

```bash
# 20 games vs the greedy bot (default)
python rl_eval.py --model checkpoints\best.pt --games 20

# Compare two checkpoints (run separately and compare win rates)
python rl_eval.py --model checkpoints\model_ep500_weights.pt  --games 20
python rl_eval.py --model checkpoints\model_ep1000_weights.pt --games 20

# Use deterministic (greedy) actions for a fair upper-bound estimate
python rl_eval.py --model checkpoints\best.pt --games 20 --deterministic

# Against idle opponent (useful early in training when bot hasn't beaten greedy yet)
python rl_eval.py --model checkpoints\model_ep100_weights.pt --games 20 --opponent idle

# Reproducible evaluation (fixed seed)
python rl_eval.py --model checkpoints\best.pt --games 20 --seed 42
```

### rl_eval.py options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to `*_weights.pt` or full `.pt` checkpoint |
| `--games` | 20 | Number of games to run |
| `--opponent` | `greedy` | `greedy`, `idle`, or `random` |
| `--deterministic` | off | Greedy actions (no sampling) — best for benchmarking |
| `--width` / `--height` | 32 | Map dimensions |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--seed` | *(random)* | Base seed (game i uses seed+i) |

### Sample output

```
Game   RL halite  Opp halite  Result  Turns  Hal/Turn
----------------------------------------------------------
    1       4230        2180      RL    400      10.6
    2       3910        3100      RL    400       9.8
    ...
Results over 20 games:
  Win rate     : 75.0%  (15W / 1T / 4L)
  Mean halite  : RL=3840  vs  Opp=2650
  Halite/turn  : 9.6
  RL advantage : +1190 halite on average
```

---

## 6. Extract imitation learning data from replays

Pre-train the bot by learning from existing replay files before running PPO.

```bash
# Extract from all replays in a folder → dataset/
python rl_collect.py replays\ --output dataset\

# Extract from a single replay
python rl_collect.py replays\replay-20260531-162050-42-40-40.hlt --output dataset\ep_001.npz
```

Each `.npz` file contains: `obs_spatial`, `obs_scalars`, `actions`, `turns`, `ship_ids`.

> **Note:** Collected data from before the rl_v2 PROSPECT update (13 scalars) is not
> compatible with the new model (17 scalars). Use a fresh dataset directory.

---

## 7. Play with the trained RL bot

### Run the RL bot against the starter-kit bot

Run from `my_extension/`:

```bash
python run_game.py \
  --bot "python rl_v2/rl_bot.py --model rl_v2/checkpoints/best.pt" \
  --bot "python ..\starter_kits\Python3\MyBot.py" \
  --replay --verbose
```

### Greedy (no exploration) mode

```bash
python run_game.py \
  --bot "python rl_v2/rl_bot.py --model rl_v2/checkpoints/best.pt --deterministic" \
  --bot "python ..\starter_kits\Python3\MyBot.py"
```

### rl_bot.py options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to `_weights.pt` file |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--deterministic` | off | Use greedy (highest prob) action instead of sampling |

---

## 8. Typical workflow

```
1. Run a few games to generate replays (from my_extension/):
   python run_game.py --replay --width 32 --height 32

2. Phase 1 — train rl_v2 against idle opponent until reward is positive (from rl_v2/):
   python rl_train.py --episodes 1000 --checkpoint-dir checkpoints --opponent-policy idle

3. Evaluate Phase 1 progress:
   python rl_eval.py --model checkpoints\model_ep1000_weights.pt --games 20 --opponent idle

4. Phase 2 — resume against greedy opponent:
   python rl_train.py --resume checkpoints\model_ep1000.pt --episodes 1000 --checkpoint-dir checkpoints --opponent-policy greedy

5. Evaluate Phase 2 progress:
   python rl_eval.py --model checkpoints\model_ep2000_weights.pt --games 20 --opponent greedy

6. Monitor training_log.csv to watch reward improve.
   Watch the 'prospect' action % — if it stays near 0%, consider increasing --ent-coef.

7. Resume if you want to keep training:
   python rl_train.py --resume checkpoints\model_ep2000.pt --episodes 1000 --checkpoint-dir checkpoints

8. Run your trained bot vs the starter-kit bot (from my_extension/):
   python run_game.py --bot "python rl_v2/rl_bot.py --model rl_v2/checkpoints/best.pt" --bot "python ..\starter_kits\Python3\MyBot.py" --replay
```

### How to read the training log

| Reward range | Meaning |
|---|---|
| Consistently negative (e.g. −200) | Bot is losing loaded ships; collision_scale penalty dominates |
| Near 0 | Bot is avoiding deaths but net cargo gain is close to zero |
| Positive and rising | Bot is collecting and depositing halite — learning is working |
| Positive then plateauing | Normal; switch to greedy opponent for a harder challenge |

---

## 9. Folder structure

```
my_extension/
├── halite_engine.py      # Shared game simulator (used by all bots)
├── run_game.py           # Shared game runner
├── replay_viewer.py      # Shared replay viewer
├── rl_v1/                # Archived – first RL bot (do not modify)
│   ├── rl_bot.py
│   ├── rl_env.py
│   ├── rl_features.py
│   ├── rl_model.py
│   ├── rl_train.py
│   ├── rl_eval.py
│   ├── rl_collect.py
│   └── checkpoints_v9/
├── rl_v2/                # Archived – exploration bot with PROSPECT action
├── rl_v3/                # Archived – dropoff-heuristic bot (benchmark opponent)
└── rl_v4/                # Active – learned dropoff, production-aligned reward
    ├── README.md         # rl_v4 overview + commands
    ├── rl_features.py    # 9 actions (adds DROPOFF), 14 channels, 29 scalars
    ├── rl_bot.py rl_env.py rl_model.py rl_train.py rl_eval.py rl_collect.py
    └── checkpoints/      # Training output (weights tracked for replication)
```
See `my_extension/CHANGELOG.md` for the change history.

