# rl_v7 — Complete Project Guide

## Goal

Build a machine-learning bot, **rl_v7**, that defeats the existing rule-based FSM bot **rl_v5** in Halite III 1v1 matches on 32×32 maps.

**Success criterion:** rl_v7 wins >50% of games against rl_v5 over 50+ random seeds.

---

## Background: What Is rl_v5?

rl_v5 is a **finite state machine (FSM) hybrid bot**. Each ship runs through four states:

- **PROSPECT** — scan a 5-cell-radius window, move toward the richest cell
- **HARVEST** — mine the current cell until cargo ≥ 90% full
- **HOME** — route home along the least-cost path (Dijkstra, not pure Manhattan)
- **ESCAPE** — break a jam when stuck for 5+ turns by wandering randomly

On top of the FSM, rl_v5 wraps a small neural network that can override the FSM suggestion when it has learned something better. It also runs a deterministic 4-phase collision resolver, spawns ships using a proven economy rule, and builds dropoffs via a learned action.

rl_v5 deposits roughly **14,000–19,000 halite per game** and is the benchmark to beat.

---

## Why Not Simply Extend rl_v5?

rl_v5 is self-contained and well-tuned. The plan is to **imitate** it first (so rl_v7 starts with rl_v5-quality play), then use **reinforcement learning** to push past it — letting RL find edges in target selection, return timing, and dropoff placement that the hand-crafted FSM misses.

---

## Concept & Approach

### Three-Phase Pipeline

```
Phase 1 — Data generation
  rl_v5 plays many games against itself → .hlt replay files

Phase 2 — Behavioral Cloning (BC)
  Parse replays → (features, action) rows in CSV format
  Train a neural net to reproduce rl_v5's decisions
  Target: val match-rate ≥ 90%  (rl_v7 plays like rl_v5)

Phase 3 — PPO Fine-Tuning
  Warm-start from BC weights
  rl_v7 plays against frozen rl_v5, collecting win/loss signal
  RL searches for strategy improvements over rl_v5's fixed heuristics
  Target: win rate > 50% over 50 games
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Collision avoidance | Deterministic resolver (not learned) | A per-ship net cannot learn joint fleet coordination; the resolver guarantees safety so RL exploration never blows up the fleet |
| Economy (spawn/dropoffs) | Replicate rl_v5's rules exactly | Removes economy as a variable; the learned ship intent is the only differentiator |
| Feature format | ~39 named scalar columns (human-readable CSV) + 9×9×6 map patch | Scalars are inspectable; patch gives spatial context for enemy/inspiration awareness |
| Network | MLP on scalars + small CNN on patch | Matches the data format; fast enough for live inference |
| Opponent during RL | Frozen rl_v5 checkpoint | Clear signal: beating the best known benchmark |
| Bot naming | `game.ready("rl_v7")` / `game.ready("rl_v5")` | Replays show correct player names (rl_v5 originally reported "RLBot") |

---

## Architecture

### Neural Network (`net.py`)

```
Scalar branch   : float32[39] → Linear(128) → ReLU → Linear(128) → ReLU  → [128]
Patch branch    : float32[6,9,9] → Conv(32,3×3) → ReLU → Conv(32,3×3) → ReLU
                                 → flatten[2592] → Linear(64) → ReLU        → [64]

Trunk           : concat[192] → Linear(256) → ReLU → Linear(128) → ReLU    → [128]

Actor head      : Linear(6)   → action logits  (STAY / N / E / S / W / DROPOFF)
Critic head     : Linear(1)   → state value    (used by PPO)
```

Action masking (DROPOFF only legal under specific economy conditions) is applied at inference as `-inf` on illegal logits.

### Deterministic Resolver (`resolver.py`)

Runs after the net produces per-ship intents:

1. **Affordability check** — if cargo < cell_halite/10, ship cannot move; force STAY
2. **Endgame force-home** — if turns_left ≤ dist_to_deposit + 5, override to least-cost home step
3. **Enemy threat zone** — each enemy's cell + 4 neighbours; ships moving into threat → STAY; ships sitting in threat try to escape sideways
4. **Factory reservation** — if spawning, prevent any friendly from moving onto factory this turn
5. **Cascade** — stayers own their cell; movers yielding into an occupied cell → STAY; heaviest-cargo mover wins ties; repeat until stable
6. **Endgame exemption** — final 15 turns: ships heading to own deposit pile on (engine banks cargo before collision)

This guarantees **zero friendly self-collisions by construction**. The net never has to learn fleet coordination.

---

## Feature Design

### Scalar Features (39 columns in `features.csv`)

| Group | Features |
|---|---|
| Per-ship economy | `cargo_frac`, `cargo_to_full_frac`, `cargo_ge_home`, `cell_halite_frac`, `mine_yield_frac`, `can_afford_move`, `is_inspired` |
| Homing | `dist_home_frac`, `dx_home`, `dy_home`, `on_deposit`, `return_urgency`, `turns_slack` |
| Prospecting | `dx_richest`, `dy_richest`, `richest_halite_frac`, `dist_richest_frac`, `local_mean3_frac`, `window_mean_frac` |
| Danger | `enemy_within_1`, `enemy_within_2`, `friendly_within_1`, `friendly_within_2`, `min_enemy_cargo_near`, `enemy_count_r4` |
| Global / fleet / phase | `turn_frac`, `turns_left_frac`, `my_ships_frac`, `opp_ships_frac`, `my_bank_frac`, `opp_bank_frac`, `bank_margin_tanh`, `winning_ratio`, `num_dropoffs_frac`, `map_halite_frac`, `dropoff_affordable`, `dropoff_legal`, `map_w_norm`, `map_h_norm` |

Each row also has meta-columns: `row_id`, `game_id`, `player_id`, `ship_id`, `turn`, `action`.

`action` values: 0=STAY, 1=NORTH, 2=EAST, 3=SOUTH, 4=WEST, 5=DROPOFF.

### Map Patch (`patches.npy`)

Shape `[N, 9, 9, 6]` (float16). Centred on the ship, radius-4 window (covers Halite's inspiration radius). Channels:

| Ch | Content |
|---|---|
| 0 | cell halite / 1000 |
| 1 | friendly ship present |
| 2 | enemy ship present |
| 3 | friendly ship cargo / 1000 |
| 4 | enemy ship cargo / 1000 |
| 5 | +1 = my factory/dropoff, −1 = enemy factory/dropoff |

`patches.npy` row `i` corresponds to `features.csv` `row_id == i`.

---

## File Structure

```
my_extension/rl_v7/
├── config.py           All constants, action enum, feature names, economy helpers
├── features.py         Single-source WorldView + feature extraction
│                         world_view_from_replay()  ← used by collect_dataset
│                         world_view_from_engine()  ← used by rl_env (PPO)
│                         world_view_from_hlt()     ← used by rl_bot (inference)
├── net.py              ActorCritic policy/value network
├── resolver.py         Deterministic collision resolver
├── generate_games.py   Run rl_v5 self-play, save .hlt replays
├── collect_dataset.py  Parse replays → features.csv + patches.npy
├── bc_train.py         Behavioral cloning training
├── rl_bot.py           Live inference bot (reports name "rl_v7")
├── v5_opponent.py      rl_v5 wrapper (reports name "rl_v5", rl_v5 folder untouched)
├── rl_env.py           PPO training environment (wraps HaliteEngine)
├── rl_train.py         PPO fine-tuning
├── rl_eval.py          Head-to-head evaluation + replay name check
├── dataset/
│   ├── features.csv    Human-readable feature rows
│   └── patches.npy     Map patches aligned by row_id
├── checkpoints/
│   ├── bc.pt           Best behavioral-cloning checkpoint
│   ├── ppo_ep####.pt   PPO checkpoints every 25 episodes
│   ├── ppo_final.pt    Final PPO checkpoint
│   └── ppo_log.csv     Per-episode training log
└── replays/            .hlt replay files
```

All commands run from `my_extension/`. rl_v5 and all other folders are **never modified**.

---

## Step-by-Step Run Commands

### Step 1 — Generate rl_v5 self-play games

```powershell
python rl_v7/generate_games.py --games 300
```

- Runs rl_v5 vs rl_v5 (stochastic; diverse states)
- Saves `.hlt` replays to `rl_v7/replays/`
- Each game takes ~10–30 seconds; 300 games ≈ 1–2 hours
- Optional: add greedy games for extra diversity

```powershell
python rl_v7/generate_games.py --games 250 --vs-greedy 50
```

**Expected output:** one line per game showing seed, halite totals, and saved filename.

---

### Step 2 — Collect the dataset

```powershell
python rl_v7/collect_dataset.py
```

- Reads all `.hlt` files in `rl_v7/replays/`
- Only collects rows from seats named `"rl_v5"` (both seats in self-play)
- Writes `rl_v7/dataset/features.csv` and `rl_v7/dataset/patches.npy`
- Runs an **alignment sanity check**: verifies that labelled moves match actual next-frame ship positions (must be ≥ 98%)

Optional: subsample the dominant STAY action to speed up training:

```powershell
python rl_v7/collect_dataset.py --stay-keep 0.5
```

**Expected output:**
```
Found 300 replays in ...
=== dataset written ===
rows         : ~1,900,000
features.csv : ...
patches.npy  : shape=(~1900000, 9, 9, 6) dtype=float16

action distribution:
  STAY    :  ~40%
  NORTH   :  ~11%
  EAST    :  ~20%
  SOUTH   :  ~13%
  WEST    :  ~17%
  DROPOFF :  ~0%

alignment sanity check: NNN/NNN moves landed as labelled (99.x%)  OK
```

**Inspect the CSV:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('rl_v7/dataset/features.csv'); print(df.shape); print(df[['cargo_frac','dist_home_frac','richest_halite_frac','action']].head(10))"
```

---

### Step 3 — Behavioral cloning

```powershell
python rl_v7/bc_train.py --epochs 30 --device cuda
```

- Class-weighted cross-entropy (inverse-frequency, capped at 10×) so rare moves are not ignored
- 90/10 train/val split; saves best checkpoint to `checkpoints/bc.pt`
- Reports per-action recall on validation set

**Expected output (epoch by epoch):**
```
epoch   1  loss 1.65  val_match 45.20%  *saved*
...
epoch  30  loss 0.42  val_match 91.30%  *saved*

best val match-rate: 91.30%  ->  checkpoints/bc.pt
per-action recall (val):
  STAY    :  96.2%
  NORTH   :  88.1%
  EAST    :  94.3%
  SOUTH   :  89.7%
  WEST    :  91.5%
```

**Target:** val match-rate ≥ ~90%.

If match-rate is below 85% after 30 epochs, generate more games (target 400–500 total) and rerun collection + training.

CPU fallback (slower):
```powershell
python rl_v7/bc_train.py --epochs 30 --device cpu
```

---

### Step 4 — Evaluate the BC bot

```powershell
python rl_v7/rl_eval.py --model rl_v7/checkpoints/bc.pt --games 20 --save-replays
```

- Runs 20 games: rl_v7 (bot) vs rl_v5 (bot)
- Reports win rate, mean halite per side, and verifies replay names = "rl_v7" / "rl_v5"

**Expected output after good BC:**
```
game    rl_v7    rl_v5  result  names-ok
   1   12,340   13,200  rl_v5   OK
   2   14,100   11,800  rl_v7   OK
...
win rate  : 8/20 = 40.0%
mean halite: rl_v7=12,800  rl_v5=13,200
```

BC alone reaching 40–50% win rate is a strong sign; PPO will push it over 50%.

View a replay:
```powershell
python replay_viewer.py rl_v7/replays/eval-v7vsv5-<seed>-<ts>.hlt
```

---

### Step 5 — PPO fine-tuning

```powershell
python rl_v7/rl_train.py --bc-ckpt rl_v7/checkpoints/bc.pt --episodes 500 --device cuda
```

- Warm-starts from the BC checkpoint
- Plays 6 games per PPO update (reduces variance)
- Entropy floor guard prevents policy collapse
- LR decays 3% every 50 episodes after episode 100
- Checkpoints every 25 episodes; logs to `checkpoints/ppo_log.csv`

**Expected output:**
```
ep    1  dep=  8200/13100  wr=0.17  ent=1.23  lr=2.00e-04  t=45.2s
...
ep   50  dep= 11400/12800  wr=0.33  ent=1.05  lr=2.00e-04  t=43.1s
  saved checkpoints/ppo_ep0050.pt
...
ep  200  dep= 13800/12600  wr=0.55  ent=0.87  lr=1.88e-04  t=42.0s
  saved checkpoints/ppo_ep0200.pt
```

**Resume from a checkpoint** (if training was interrupted):
```powershell
python rl_v7/rl_train.py --resume rl_v7/checkpoints/ppo_ep0200.pt --start-ep 201 --episodes 300
```

**Monitor training** (inspect the log):
```powershell
python -c "import pandas as pd; df = pd.read_csv('rl_v7/checkpoints/ppo_log.csv'); print(df[['episode','deposited_agent','deposited_opp','win_rate','mean_entropy']].tail(20).to_string())"
```

Watch for:
- `win_rate` climbing above 0.5
- `mean_entropy` staying above ~0.3 (collapse prevention working)
- `deposited_agent` approaching or exceeding `deposited_opp`

---

### Step 6 — Final evaluation

```powershell
python rl_v7/rl_eval.py --model rl_v7/checkpoints/ppo_final.pt --games 50 --save-replays
```

**Target output:**
```
win rate  : 28/50 = 56.0%
mean halite: rl_v7=14,200  rl_v5=13,100

*** rl_v7 beats rl_v5! (56.0% win rate) ***
```

If not yet at 50%, run more PPO episodes:
```powershell
python rl_v7/rl_train.py --resume rl_v7/checkpoints/ppo_final.pt --start-ep 501 --episodes 250
```

---

### Quick single game (watch live in viewer)

```powershell
python run_game.py `
  --bot "python -u rl_v7/rl_bot.py --model rl_v7/checkpoints/ppo_final.pt --deterministic" `
  --bot "python -u rl_v7/v5_opponent.py --deterministic" `
  --replay --verbose
```

Then open the replay:
```powershell
python replay_viewer.py rl_v7/replays/replay-<timestamp>-<seed>-32-32.hlt
```

---

## Tuning Guide

### If BC match-rate is low (< 85% after 30 epochs)

- Generate more games: increase `--games` to 400–500
- Try `--stay-keep 0.5` to balance STAY-heavy data
- Try more epochs: `--epochs 50`

### If PPO doesn't improve win rate

- Check entropy isn't collapsing (should stay > 0.3): `ppo_log.csv` → `mean_entropy` column
- Try more games per update: `--games-per-update 10`
- Lower LR: `--lr 5e-5`
- Try a mid-training checkpoint rather than final (PPO can overshoot)

### If rl_v7 has many self-collisions

The resolver guarantees zero friendly self-collisions. If you see collision events in replays between friendly ships, the resolver has a bug — open a bug against `resolver.py`.

---

## Implementation Notes

### Zero train/inference skew

`features.py` contains three adapters — `world_view_from_replay()`, `world_view_from_engine()`, `world_view_from_hlt()` — that all produce the same `WorldView` object and call the same `extract_scalars()` / `extract_patch()` functions. The network always sees the same features regardless of whether it is being trained or deployed.

### The `_apply` / `_mask_and_prior` gotcha

PyTorch's `nn.Module` uses an internal method named `_apply()` for device/dtype conversion (called by `.to()`, `.cuda()`, etc.). Any custom method named `_apply` in a subclass **shadows this and breaks `.to()`**. In `net.py` the masking helper is therefore named `_mask_and_prior`.

### rl_v5 folder is never modified

`v5_opponent.py` monkeypatches `hlt.Game.ready` at runtime so the engine records the player name as `"rl_v5"` without touching any file in `rl_v5/`. All original bot folders remain pristine archives.

---

## Experiment Results & Conclusions

### What was run

Four PPO training runs were attempted. Run 4 (the final and most stable) used BC regularisation (KL penalty) and a conditional mine reward:

```powershell
python rl_v7/rl_train.py \
  --bc-ckpt rl_v7/checkpoints/bc.pt \
  --episodes 100 --device cuda \
  --lambda-bc 1.0 --lambda-bc-min 0.05 --lambda-bc-episodes 200
```

### Actual results (Run 4)

| Checkpoint | rl_v7 avg deposits | rl_v5 avg deposits | Win rate |
|-----------|-------------------|-------------------|----------|
| `bc.pt` | ~10–12k | ~14k | ~20–30% |
| `ppo_ep0050.pt` | ~10k | ~14k | ~25% |
| `ppo_ep0100.pt` / `ppo_final.pt` | ~7.6k | ~13k | ~14% |

**BC match-rate: ~58.5%** (project guide estimated 90%+ was achievable — see structural diagnosis below for why it was not).

The KL regularisation successfully prevented the hard policy collapse seen in Runs 1–3 (entropy → 0.3 floor, deposits → 1–3k within 50 episodes). However deposits still drifted down from ~11k → ~7–8k over 100 episodes.

### Root cause: structural gap between rl_v5 and rl_v7

rl_v7 is a **stateless per-turn policy** trying to imitate a **stateful FSM hybrid**. Four structural gaps cannot be bridged by a single-frame feature vector:

| Capability | rl_v5 | rl_v7 |
|-----------|-------|-------|
| Per-ship persistent state | FSM (HARVEST/HOME/ESCAPE + target lock-on) | None — re-decides every turn |
| Fleet target deconfliction | `_claimed` set; ships spread naturally | Implicit only (patch CNN) |
| Jam-breaking | ESCAPE after 5 stuck turns | None |
| Endgame collapse | Final 15 turns: pile ships onto factory | Resolver serialises deposits |
| Structural prior | +3.0 logit anchors NN to FSM suggestion | Pure NN, no bias |

The 58.5% BC ceiling is direct evidence: a ship's FSM state (HARVEST vs PROSPECT) is not observable from a single frame, so the model cannot predict the correct action.

### Why wins for rl_v7 happened

rl_v7 wins occurred when rl_v5 scored anomalously low (2–6k vs its normal 14–20k), not when rl_v7 outplayed it. Best representative win replays:

| Replay | rl_v7 | rl_v5 | Note |
|--------|-------|-------|------|
| `eval-v7vsv5-1000-073617.hlt` | 8,500 | 5,148 | rl_v5 bad map |
| `eval-v7vsv5-1005-073920.hlt` | 6,904 | 2,394 | Most dominant win |
| `eval-v7vsv5-1004-073841.hlt` | 7,272 | 6,528 | Closest competitive win |

### Recommended next steps to beat rl_v5

1. **Add shadow FSM to rl_v7** (`rl_env.py` + `rl_bot.py`): run a per-ship FSM and expose PROSPECT/HARVEST/HOME/ESCAPE as scalar features. This directly lifts the 58.5% BC ceiling — the model can now predict stateful transitions.
2. **Add home-memory flag**: sticky `homing_ships` set (cleared on deposit), mirroring rl_v5's hard home-commitment override.
3. **BC on augmented features first** (Steps 1–3 again): rerun dataset collection with the shadow FSM features, retrain BC. Target ≥ 80% match-rate before resuming PPO.
