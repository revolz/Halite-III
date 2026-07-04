# rl_v8  --  imitation + RL bot trained on the 2019 bot (MyBot V71)

## Goal
Beat both `rl_v5` and `rl_v7` in 1v1 Halite III matches on 32x32 maps, by
learning from the user's old hand-coded 2019 bot, `Year 2019/MyBot - V71`,
which swept a head-to-head pitch against both current RL bots (9-1 combined
vs rl_v5, 10-0 vs rl_v7 across two match sets).

**Published checkpoint:** `checkpoints/best.pt` is the final **BC** policy
(the file the pipeline below writes as `bc.pt`) — rl_v8's strongest, because
its PPO run degraded the BC policy rather than improving it (the failure
rl_v9 was built to diagnose). PPO snapshots stay local.

## Design note: this is the "pure stateless BC" variant, on purpose
Before building this, we read V71's full source (`MyBot.py`, 739 lines) and
found it is **not** a simple stateless heuristic -- it's a per-ship finite
state machine (`SEARCHING -> NAVIGATING -> MINING -> RETURNING`, plus an
endgame override), with genuinely hidden turn-to-turn state (`self.state`,
a queued path buffer `self.mcp_step`) persisted in a `revolz_fleet` dict
(`MyBot.py:640,717-718,729-730`).

This is the same category of problem that capped `rl_v7`'s imitation of
`rl_v5` (also an FSM) at ≈58% behavioral-cloning match rate and caused it to
lose every match (see `rl_v7/README.md`, "Status & Conclusions"). **The owner
explicitly chose to build rl_v8 without a shadow-FSM feature fix**, to test
whether the same ceiling reappears against a different FSM-based target bot.
If BC plateaus low here too, that's the expected/tested result, not a bug --
a shadow-FSM feature block (re-deriving V71's SEARCHING/NAVIGATING/MINING/
RETURNING state from public board state, fed as extra input features) is the
documented fallback if this gets revisited.

## Architecture comparison: V71 vs rl_v8

rl_v8 is **not a purely-RL bot**. Only the per-ship move/dropoff choice is
learned (via BC then PPO); spawn timing, dropoff legality, and collision
safety are all deterministic hand-coded rules, same as every other bot in
this repo:

| Component | Learned? |
|---|---|
| Per-ship move/dropoff choice | **Yes** -- the `ActorCritic` net, trained via BC then PPO |
| Feature representation (39 scalars + 9x9x6 patch) | No -- hand-engineered |
| Spawn timing | No -- `config.spawn_econ_ok()`, ported from rl_v5, not V71 |
| Dropoff legality | No -- `config.dropoff_legal()`, fixed rule |
| Collision safety | No -- `resolver.py`, deterministic cascade overriding the net's intents on conflict |

| Aspect | V71 (2019) | rl_v8 |
|---|---|---|
| Decision mechanism | Hand-coded finite state machine (`SEARCHING -> NAVIGATING -> MINING -> RETURNING`) | Neural network (small MLP+CNN `ActorCritic`), no explicit states |
| Per-ship memory | Persistent across turns -- `self.state`, a queued path (`self.mcp_step`), stored in a module-level `revolz_fleet` dict | **None** -- stateless, recomputes everything fresh from the board every turn (deliberate design choice, see above) |
| Targeting logic | Explicit ROI formula (expected value minus decay minus path cost) scanned over a local radius-3 window (`create_roi_map`) | Implicit -- learned end-to-end from the 9x9 local patch + scalars, no formula a human wrote |
| Path-following | Precomputes a multi-step minimum-cost path (`mcp_step`, a FILO stack from `min_cost_path`/`min_cost_path_step`), pops one step per turn | No path commitment -- picks one action per turn independently every time |
| Collision avoidance | `unsafe_map`, ships processed sequentially, each one's destination marked unsafe for the next | `resolver.py` -- same spirit (sequential, deterministic), separate generic implementation, not V71's code |
| Spawn/dropoff rule | V71's own thresholds (ship-count <= richest-enemy+5; bank > 5000 & turn > 45% for dropoff) | **Different** rules entirely -- rl_v5's `spawn_econ_ok`/`target_dropoffs` formulas, not V71's, not learned |
| How it was built | Written once by hand in 2019, never trained | BC (supervised imitation of V71's replay actions) -> PPO (reward-driven fine-tuning vs rl_v5/rl_v7) |
| Can it keep improving? | No -- fixed forever | Yes -- more PPO episodes can keep changing behavior, and it can (and likely will) drift away from V71's original style since PPO optimizes reward, not imitation fidelity |
| Determinism | Deterministic except a `random_move()` fallback when stuck | Deterministic only with `--deterministic` (greedy); default mode samples stochastically |

Note: "learning from V71" is narrower than it sounds -- the BC dataset only
covers V71's *movement/mining/homing* style. rl_v8's *economy* (spawn/dropoff
timing) was never learned from V71 at all; it silently inherited rl_v5's
economy rules instead (carried over from the rl_v7 scaffolding).

MCP = **Minimum Cost Path** (V71's own hand-rolled weighted-shortest-path
routing, conceptually the same idea rl_v5 later implemented as
`least_cost_home_step`/`compute_home_cost_field`, written independently in
2019). `self.mcp_step` is the precomputed move sequence, stored as a stack
("last one to execute first ... FILO", per V71's own code comments) and
popped one step per turn -- this is exactly the persistent hidden state a
stateless imitator like rl_v8 cannot directly observe.

## Pipeline

```
V71-vs-rl_v5 games -> features.csv + patches.npy -> BC training
                                                          |
                                              net (MLP+CNN) imitates V71
                                                          |
                                       PPO fine-tune vs frozen rl_v5 / rl_v7
                                                          |
                                    rl_eval: win rate vs rl_v5 and vs rl_v7
```

Almost the entire pipeline is reused near-verbatim from `rl_v7/` (its
feature extraction, network, resolver, and BC trainer are bot-agnostic --
they only read public replay/engine state, never opponent-internal info):

| File | Status |
|------|--------|
| `config.py` | copied verbatim (generic action space + economy helpers) |
| `features.py` | copied verbatim (generic `WorldView`-based extraction) |
| `net.py` | copied verbatim (`ActorCritic` MLP+CNN) |
| `resolver.py` | copied verbatim (deterministic collision resolver) |
| `bc_train.py` | copied verbatim (class-weighted BC trainer) |
| `generate_games.py` | retargeted: runs **V71 vs rl_v5** by default (not V71 self-play -- pushes V71 into a wider variety of board states than it creates on its own; rl_v5 runs stochastically for extra diversity). `--selfplay`/`--vs-greedy` add other sources |
| `collect_dataset.py` | retargeted: collects the seat named `"RevolzBot"` (V71's `game.ready()` name) instead of `"rl_v5"` |
| `rl_bot.py` | renamed: reports `"rl_v8"`, otherwise identical inference logic |
| `rl_env.py` | generalized: `FrozenOpponent` drives *any* subprocess bot command (was hardcoded to rl_v5 via a wrapper script) -- discovered this was already just a subprocess pipe, no in-process PyTorch loading needed |
| `rl_train.py` | added `--opponent {rl_v5,rl_v7,alternate}` (default `alternate`, round-robins both per PPO update) so rl_v8 doesn't overfit to beating just one bot |
| `rl_eval.py` | generalized to benchmark vs both rl_v5 and rl_v7 in one run |

V71 has no in-process Python API and no `--deterministic` flag -- it always
runs as its own subprocess, exactly like any other bot on the standard hlt
protocol. Economy (spawn/dropoff thresholds) is **not** learned from V71;
it reuses `config.py`'s generic rules (already ported from rl_v5 by rl_v7),
matching rl_v7's original design choice.

## Step-by-step run commands

All commands are run from `my_extension/`.

### 1. Generate dataset (V71 vs rl_v5, default)

```powershell
python rl_v8/generate_games.py --games 300
```

Add extra state diversity (V71 self-play and/or vs the starter-kit bot):
```powershell
python rl_v8/generate_games.py --games 250 --selfplay 30 --vs-greedy 20
```

### 2. Collect features

```powershell
python rl_v8/collect_dataset.py
```

To sub-sample the dominant STAY action (optional, speeds up training):
```powershell
python rl_v8/collect_dataset.py --stay-keep 0.5
```

### 3. Behavioral cloning

```powershell
python rl_v8/bc_train.py --epochs 30 --device cuda
```

Report the val match-rate plainly -- this is the specific hypothesis being
tested (does imitating a different FSM bot also plateau around ≈55-60%?).

### 4. Evaluate BC bot vs rl_v5 and rl_v7

```powershell
python rl_v8/rl_eval.py --model rl_v8/checkpoints/bc.pt --games 20
```

### 5. PPO fine-tuning (vs both opponents, alternating)

```powershell
python rl_v8/rl_train.py --bc-ckpt rl_v8/checkpoints/bc.pt --opponent alternate --episodes 500 --device cuda
```

Train vs a single opponent instead:
```powershell
python rl_v8/rl_train.py --bc-ckpt rl_v8/checkpoints/bc.pt --opponent rl_v5 --episodes 500 --device cuda
```

Resume from a checkpoint:
```powershell
python rl_v8/rl_train.py --resume rl_v8/checkpoints/ppo_ep0200.pt --start-ep 201 --episodes 300
```

### 6. Final evaluation

```powershell
python rl_v8/rl_eval.py --model rl_v8/checkpoints/ppo_final.pt --games 50 --save-replays
```

Target: win rate > 50% vs **both** rl_v5 and rl_v7.

### Quick head-to-head via run_game.py (watch one game)

```powershell
python run_game.py `
  --bot "python -u rl_v8/rl_bot.py --model rl_v8/checkpoints/ppo_final.pt --deterministic" `
  --bot "python -u rl_v5/rl_bot.py --model rl_v5/checkpoints/best.pt --deterministic" `
  --replay --verbose
```

Open the replay in the viewer:
```powershell
python replay_viewer.py replays/<filename>.hlt
```
