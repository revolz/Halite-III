# Halite III – Command Reference

All commands are run from the `my_extension/` directory unless stated otherwise.

```bash
cd "C:\Temp\Halite-III - 02 - Second Version\my_extension"
```

---

## 1. Run a game

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

### Run your own bot

```bash
# Your bot vs the starter-kit bot
python run_game.py --bot "python MyBot.py" --bot "python ..\starter_kits\Python3\MyBot.py"
```

---

## 2. Watch a replay

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

### Recommended 3-phase training schedule

Train in phases — start easy, then increase difficulty:

**Phase 1 — Learn to mine (idle opponent, 1000 episodes)**
```bash
python rl_train.py --episodes 1000 --checkpoint-dir checkpoints --opponent-policy idle
```
Expected: reward goes from ~0 to positive as bot learns to collect and deposit halite.

**Phase 2 — Add competition (greedy opponent, resume for 1000 more)**
```bash
python rl_train.py --resume checkpoints\model_ep1000.pt --start-episode 1001 --episodes 1000 --checkpoint-dir checkpoints --opponent-policy greedy
```
Expected: reward may dip briefly then recover as bot adapts to an opponent.

**Phase 3 — Fine-tune (optional)**
```bash
python rl_train.py --resume checkpoints\model_ep2000.pt --start-episode 2001 --episodes 1000 --checkpoint-dir checkpoints --opponent-policy greedy
```

### Start training from scratch (single command, idle opponent)

```bash
python rl_train.py --episodes 2000 --checkpoint-dir checkpoints
```

**What happens:**
- Prints progress every 10 episodes: `Episode   10 | reward    0.0 | avg100    0.0 | 12s`
- Saves a checkpoint every 50 episodes: `checkpoints\model_ep50.pt`
- Saves plain weights alongside: `checkpoints\model_ep50_weights.pt`
- Writes a CSV log: `checkpoints\training_log.csv`
- Saves final model when done: `checkpoints\model_final.pt` + `model_final_weights.pt`

### Resume training from a checkpoint

```bash
# Stopped at episode 100, continue for 1000 more episodes
python rl_train.py --resume checkpoints\model_ep100.pt --start-episode 101 --episodes 1000 --checkpoint-dir checkpoints
```

- `--resume` loads weights **and** optimizer state (training momentum is preserved)
- `--start-episode` keeps the episode counter and checkpoint filenames consistent
- The CSV log is **appended**, so the full history stays in one file

### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 2000 | Number of episodes to train |
| `--checkpoint-dir` | `checkpoints` | Folder for checkpoints and log |
| `--checkpoint-interval` | 50 | Save a checkpoint every N episodes |
| `--resume` | *(none)* | Path to `.pt` file to resume from |
| `--start-episode` | 1 | Episode counter start (use with `--resume`) |
| `--opponent-policy` | `idle` | `idle` (no opponent moves) or `greedy` (scripted heuristic bot) |
| `--death-penalty-scale` | 0.5 | Multiplier on cargo lost when a ship dies (proportional, not flat) |
| `--cargo-reward-scale` | 0.3 | Weight on per-turn cargo-gain reward (dense learning signal) |
| `--width` / `--height` | 32 | Map dimensions |
| `--lr` | 3e-4 | Learning rate |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--seed` | *(random)* | Fixed seed for reproducible maps |

---

## 4. Check training progress

### Console output (live)
The training script prints every 10 episodes:
```
Episode    10 | reward      0.0 | avg100      0.0 | 12s
Episode    20 | reward    150.0 | avg100    80.0 | 24s
```

### CSV log (any time, including after training)
Open `checkpoints\training_log.csv` in Excel, or tail it in a second terminal:
```powershell
# PowerShell: watch the last 5 lines of the log live
while ($true) { Get-Content checkpoints\training_log.csv | Select-Object -Last 5; Start-Sleep 10 }
```

Columns: `episode`, `reward`, `avg100_reward`, `elapsed_sec`

---

## 5. Extract imitation learning data from replays

Pre-train the bot by learning from existing replay files before running PPO.

```bash
# Extract from all replays in a folder → dataset/
python rl_collect.py replays\ --output dataset\

# Extract from a single replay
python rl_collect.py replays\replay-20260531-162050-42-40-40.hlt --output dataset\ep_001.npz
```

Each `.npz` file contains: `obs_spatial`, `obs_scalars`, `actions`, `turns`, `ship_ids`.

---

## 6. Play with the trained RL bot

### Run the RL bot against the starter-kit bot

```bash
python run_game.py \
  --bot "python rl_bot.py --model checkpoints\model_final_weights.pt" \
  --bot "python ..\starter_kits\Python3\MyBot.py" \
  --replay --verbose
```

### Greedy (no exploration) mode

```bash
python run_game.py \
  --bot "python rl_bot.py --model checkpoints\model_final_weights.pt --deterministic" \
  --bot "python ..\starter_kits\Python3\MyBot.py"
```

### rl_bot.py options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to `_weights.pt` file |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--deterministic` | off | Use greedy (highest prob) action instead of sampling |

---

## 7. Typical workflow

```
1. Run a few games to generate replays:
   python run_game.py --replay --width 32 --height 32

2. (Optional) Extract imitation data from good replays:
   python rl_collect.py replays\ --output dataset\

3. Phase 1 — train against idle opponent until reward is positive:
   python rl_train.py --episodes 1000 --checkpoint-dir checkpoints --opponent-policy idle

4. Phase 2 — resume against greedy opponent:
   python rl_train.py --resume checkpoints\model_ep1000.pt --start-episode 1001 --episodes 1000 --checkpoint-dir checkpoints --opponent-policy greedy

5. Monitor training_log.csv to watch reward improve.

6. Test a checkpoint against the scripted bot:
   python run_game.py --bot "python rl_bot.py --model checkpoints\model_ep500_weights.pt" --bot "..\starter_kits\Python3\MyBot.py" --verbose

7. Resume if you want to keep training:
   python rl_train.py --resume checkpoints\model_ep2000.pt --start-episode 2001 --episodes 1000 --checkpoint-dir checkpoints
```

### How to read the training log

| Reward range | Meaning |
|---|---|
| Consistently negative (e.g. −200) | Bot is losing loaded ships; `death_penalty_scale` may be too high |
| Near 0 | Bot is avoiding deaths but not mining much yet |
| Positive and rising | Bot is collecting and depositing halite — learning is working |
| Positive then plateauing | Normal; switch to greedy opponent for harder challenge |
