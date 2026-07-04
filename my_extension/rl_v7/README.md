# rl_v7  --  Imitation + RL bot to defeat rl_v5

## Goal
Beat rl_v5 in a 1v1 Halite III match on 32×32 maps (>50% win rate).

## Pipeline

```
rl_v5 self-play games → features.csv + patches.npy → BC training
                                                          ↓
                                              net (MLP+CNN) imitates rl_v5
                                                          ↓
                                              PPO fine-tune vs frozen rl_v5
                                                          ↓
                                               rl_eval: win rate > 50%
```

## File overview

| File | Purpose |
|------|---------|
| `config.py` | All constants, action enum, feature names, economy helpers |
| `features.py` | Single-source feature extraction (scalars + 9×9×6 patch) |
| `net.py` | Policy/value network (MLP branch + CNN patch branch) |
| `resolver.py` | Deterministic collision resolver (no self-collisions by construction) |
| `generate_games.py` | Run rl_v5 self-play, save .hlt replays |
| `collect_dataset.py` | Parse replays → features.csv + patches.npy |
| `bc_train.py` | Behavioral cloning |
| `rl_bot.py` | Live inference bot (reports name "rl_v7") |
| `v5_opponent.py` | rl_v5 bot wrapper (reports name "rl_v5") |
| `rl_env.py` | PPO training environment |
| `rl_train.py` | PPO fine-tuning |
| `rl_eval.py` | Head-to-head evaluation vs rl_v5 |

## Step-by-step run commands

All commands are run from `my_extension/`.

### 1. Generate dataset (300 games ≈ 1.9M rows)

```powershell
python rl_v7/generate_games.py --games 300
```

Add extra state diversity:
```powershell
python rl_v7/generate_games.py --games 250 --vs-greedy 50
```

### 2. Collect features

```powershell
python rl_v7/collect_dataset.py
```

Inspect the output:
```powershell
python -c "import pandas as pd; df = pd.read_csv('rl_v7/dataset/features.csv'); print(df.shape); print(df.head())"
```

To sub-sample the dominant STAY action (optional, speeds up training):
```powershell
python rl_v7/collect_dataset.py --stay-keep 0.5
```

### 3. Behavioral cloning

```powershell
python rl_v7/bc_train.py --epochs 30 --device cuda
```

Target: val match-rate ≥ 90%.

### 4. Evaluate BC bot vs rl_v5

```powershell
python rl_v7/rl_eval.py --model rl_v7/checkpoints/bc.pt --games 20 --save-replays
```

### 5. PPO fine-tuning

```powershell
python rl_v7/rl_train.py --bc-ckpt rl_v7/checkpoints/bc.pt --episodes 500 --device cuda
```

Resume from a checkpoint:
```powershell
python rl_v7/rl_train.py --resume rl_v7/checkpoints/ppo_ep200.pt --start-ep 201 --episodes 300
```

### 6. Final evaluation

```powershell
python rl_v7/rl_eval.py --model rl_v7/checkpoints/ppo_final.pt --games 50 --save-replays
```

Target: win rate > 50%.

### Quick head-to-head via run_game.py (watch one game)

```powershell
python run_game.py `
  --bot "python -u rl_v7/rl_bot.py --model rl_v7/checkpoints/ppo_final.pt --deterministic" `
  --bot "python -u rl_v7/v5_opponent.py --deterministic" `
  --replay --verbose
```

Open the replay in the viewer:
```powershell
python replay_viewer.py rl_v7/replays/<filename>.hlt
```

## Feature columns (features.csv)

| Group | Columns |
|-------|---------|
| Per-ship economy | cargo_frac, cargo_to_full_frac, cargo_ge_home, cell_halite_frac, mine_yield_frac, can_afford_move, is_inspired |
| Homing | dist_home_frac, dx_home, dy_home, on_deposit, return_urgency, turns_slack |
| Prospecting | dx_richest, dy_richest, richest_halite_frac, dist_richest_frac, local_mean3_frac, window_mean_frac |
| Danger | enemy_within_1/2, friendly_within_1/2, min_enemy_cargo_near, enemy_count_r4 |
| Global | turn_frac, turns_left_frac, my/opp_ships_frac, my/opp_bank_frac, bank_margin_tanh, winning_ratio, num_dropoffs_frac, map_halite_frac, dropoff_affordable, dropoff_legal, map_w/h_norm |

## Map tensor (patches.npy)

Shape `[N, 9, 9, 6]` (float16). Channel meanings:

| Channel | Content |
|---------|---------|
| 0 | cell halite / 1000 |
| 1 | friendly ship present |
| 2 | enemy ship present |
| 3 | friendly ship cargo / 1000 |
| 4 | enemy ship cargo / 1000 |
| 5 | +1 my deposit, -1 enemy deposit |

---

## Status & Conclusions (Project Concluded)

**rl_v7 did not achieve the >50% win rate target against rl_v5.**

### Actual results (PPO Run 4 — BC reg + conditional reward)

| Phase | rl_v7 deposits | rl_v5 deposits | Win rate |
|-------|---------------|---------------|----------|
| BC checkpoint | ~10–12k | ~14k (opp) | ~20–30% |
| PPO ep 50 | ~10k | ~14k | ~25% |
| PPO ep 100 | ~7.6k | ~13k | ~14% |

BC match-rate ceiling: **~58.5%** (GUIDE.md projected 90% — the gap is the structural diagnosis below).

### Why rl_v7 could not beat rl_v5

rl_v5 is a stateful FSM hybrid; rl_v7 is a stateless per-turn policy. Four structural gaps explain the shortfall:

1. **Persistent per-ship state**: rl_v5's `FSMController` tracks PROSPECT/HARVEST/HOME/ESCAPE across turns — ships commit to targets and homing routes. rl_v7 re-decides every turn with no memory; loaded ships can "get distracted" mid-return, and stuck ships cannot break out.

2. **Fleet coordination via target deconfliction**: rl_v5's `_claimed` set prevents two ships targeting the same rich cell. rl_v7 must learn spreading from proximity features alone — a harder implicit coordination problem.

3. **Endgame collapse**: In the final 15 turns rl_v5 piles multiple ships onto the factory simultaneously to cash out cargo (engine credits before the wreck). rl_v7's resolver serialises deposits — ships queue instead.

4. **FSM logit prior**: rl_v5 adds +3.0 to the FSM-suggested action's logit, anchoring behaviour without training. rl_v7 is purely reactive with no structural bias.

The 58.5% BC ceiling itself proves the core problem: a stateless single-frame feature vector cannot predict a stateful FSM — the model cannot observe whether a ship is in HARVEST or PROSPECT state.

PPO with KL regularisation stabilised entropy but could not learn what the features cannot express. Deposits drifted down from ~11k → ~7-8k over 100 episodes.

### Potential next steps (not implemented)

- **Add shadow FSM to rl_v7**: run a per-ship FSM in `rl_env.py` and `rl_bot.py`, expose PROSPECT/HARVEST/HOME/ESCAPE as scalar features. Directly lifts the 58.5% BC ceiling.
- **Add home-memory flag**: sticky "this ship is going home" flag cleared on deposit — mirrors rl_v5's `homing_ships` set.
- **Best checkpoint for replay**: `checkpoints/ppo_ep0050.pt` produced the most competitive play (wins on seeds 1000, 1004, 1005 — rl_v7 scoring 7–9k vs rl_v5 scoring 2–6k on those maps). It is published in the repo as `checkpoints/best.pt` (the repo-wide convention: one strongest checkpoint per bot; other snapshots stay local).
