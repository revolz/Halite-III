# rl_v9 — learn everything, beat V71

*(For the full rl_v1→v9 lineage — each version's objective, outcome, and
lesson — see [`../README.md`](../README.md).)*

## Goal
Beat **V71** (`Year 2019/MyBot - V71`), the user's strongest 2019 hand-coded
bot, in 1v1 on 32x32 maps (>50% win rate), with a bot that **learns all of its
decisions**: per-ship moves, dropoff construction, and ship spawning.
Only fleet hygiene (friendly-collision deconfliction, endgame recall) stays
hand-coded.

## Why rl_v8 failed and what rl_v9 changes

rl_v8's own PPO log (`rl_v8/checkpoints/ppo_log.csv`) shows the BC policy at
episode 1 **winning 67%** of games vs V71 (14.2k vs 11.6k deposits) — and PPO
then steadily destroying it (by ep200: ≈14k vs ≈24k, ≈20% wins).  The owner's
observation "BC is a good start, PPO always makes it worse" is exactly what
the data shows.  Diagnosed causes and fixes:

| # | rl_v8 defect | rl_v9 fix |
|---|---|---|
| 1 | **GAE across interleaved ships**: all ships' steps were flattened turn-major and `values[t+1]` in the GAE recursion belonged to a *different ship* — advantages were largely noise | Trajectories are per-ship; GAE runs over each ship's own sequence (`rl_env.py` returns `dict sid -> steps`, `rl_train.gae_sequence`) |
| 2 | **Untrained critic shared a trunk with the BC policy**: the first PPO updates at lr 2e-4 pushed garbage value gradients through the policy trunk (rl_v8 ep2: deposits crashed 14k→3.9k) | Separate `ShipValue`/`SpawnValue` networks (no shared trunk) **plus** a value-only warmup phase (default 15 episodes) before any policy gradient |
| 3 | Ships that died mid-game never got `done=1` (bootstrap leaked across death) | Every ship sequence ends `done=1` on death/conversion/game end |
| 4 | No guard on policy drift per update | KL early stopping (`TARGET_KL=0.02`) + BC-anchor KL regularisation, annealed 1.0 → 0.1 over 150 episodes |
| 5 | Degradation was silent; final checkpoint = last checkpoint | **Eval gating**: every 20 episodes, deterministic games vs V71; `best.pt` is only ever replaced by a strictly better (win-rate, margin) score. Baseline eval of BC runs first, so PPO can never end worse than BC unnoticed |
| 6 | **Stateless policy vs an FSM target**: V71 keeps hidden per-ship state (RETURNING + a queued min-cost path), capping BC match rate at ≈58% | `FleetMemory` features: sticky homing flag (set at cargo ≥ 0.8, cleared on deposit — mirrors V71's `enough=0.8`), previous executed action (one-hot), stuck counter. Reproduced identically in replay collection, the RL env, and the live bot |
| 7 | Hand-coded spawn rule with `MAX_FLEET=16` cap | **Learned spawn**: a `SpawnPolicy` net (18 global scalars + factory-centred 8x8x4 coarse map) BC-trained on V71's own `g` commands, PPO fine-tuned. Only mask: `bank >= 1000` |
| 8 | Dropoffs gated by rl_v5 heuristics; PPO had no reward for them (never built any) | **Learned dropoff**: mask is physical only (cell unowned + affordable). BC learns V71's timing; PPO prices the construct honestly (reward = actual bank delta) and the payoff arrives as deposit rewards at the new dropoff |
| 9 | Resolver forced ships to STAY near enemies / flee enemy-adjacent cells — enemy collisions were impossible, traffic jams unresolvable | Resolver is **friendly-only**. Enemy interactions are the policy's decision; the reward prices a wreck as `(enemy cargo + enemy hull value) − (own cargo + own hull value)`, hull value = `1000·min(1, turns_left/200)`. Wrecking on your own structure banks cargo (endgame pile-on) and is credited as a deposit |

## Architecture

```
per-ship:  scalars(46: economy/homing/prospect/danger/global + MEMORY)
           + 9x9x6 local patch (CNN 32-32 -> 64)
           + 8x8x4 ship-centred coarse global map (CNN 16-16 -> 32)
           -> trunk 256 -> 128 -> 6 logits           (ShipPolicy)
           (identical body -> value scalar)          (ShipValue, PPO only)

per-turn:  spawn scalars(18) + factory-centred 8x8x4 global map
           -> 2 logits (no-spawn / spawn)            (SpawnPolicy)
           (identical body -> value scalar)          (SpawnValue, PPO only)
```

Checkpoints are bundles (`net.save_bundle`): `{ship, spawn, ship_value?, spawn_value?}`.

## Rewards (× 0.01)

* ship: `+1.0·deposited` `+0.05·mined (while cargo<50%)` `+ dropoff bank
  delta` `+ collision trade` `± 200 win ± 400·tanh(margin/3000)` on the last
  step of every ship's sequence.
* spawn stream: `+0.25·team deposits that turn` `−1000 if spawned` `+ the same
  terminal bonus`.

## Pipeline / commands (from `my_extension/`)

```powershell
# 1. data: V71 vs rl_v5 (stochastic) + V71 self-play
python rl_v9/generate_games.py --games 170 --selfplay 30

# 2. datasets (ship rows + spawn rows, memory features reconstructed)
python rl_v9/collect_dataset.py

# 3. behavioral cloning (both heads)
python rl_v9/bc_train.py --epochs 40 --device cuda

# 4. BC sanity eval
python rl_v9/rl_eval.py --model rl_v9/checkpoints/bc.pt --games 20

# 5. PPO vs V71 (value warmup -> gated PPO; best.pt = best evaluated ever)
python rl_v9/rl_train.py --episodes 300 --device cuda

# 6. final verdict
python rl_v9/rl_eval.py --model rl_v9/checkpoints/best.pt --games 50 --save-replays
```

## Results

*(filled in as runs complete)*

* Smoke BC (2 games, 2 epochs): ship val match **68.4%** — already above
  rl_v8's ≈58.5% ceiling from 300 games/30 epochs, confirming the memory
  features unlock the FSM-hidden-state plateau.

### Run log (2026-07-03)

**Data — 200 replays** (170 V71-vs-rl_v5 stochastic + 30 V71 self-play, 32x32).
The first generation run died silently at 67 games (≈3:37 AM; `gen_games.log`
empty — machine sleep or killed process suspected, not a crash) and was topped
up with `--games 103 --selfplay 30`.

**Dataset** (`collect_dataset.py`): **1,489,483 ship rows / 62,698 spawn rows**;
46 scalars, patch (6,9,9), global (4,8,8).

| action | share |
|---|---|
| STAY | 73.8% |
| N/E/S/W | 7.0 / 5.4 / 6.8 / 7.0% |
| DROPOFF | 179 rows (0.0%) |

Spawn labels: YES 8.9% (5,561 rows). Alignment sanity check 7818/7818 =
**100.0% OK**. STAY dominance is handled by class-weighted CE in `bc_train.py`
(cap 10x) — no `--stay-keep` downsampling needed. DROPOFF is inherently thin
(≈1 dropoff per game seat); even weighted it may train soft — PPO's honest
dropoff reward is the intended backstop.

**BC (40 epochs, cuda)** — the FleetMemory features didn't just crack the
FSM-hidden-state plateau, they nearly closed it:

| Run | Data | Ship val match |
|---|---|---|
| rl_v8 BC (no memory features) | 300 games, 30 epochs | ≈58.5% (the FSM ceiling) |
| rl_v9 smoke BC | 2 games, 2 epochs | 68.4% |
| **rl_v9 full BC** | 200 games, 40 epochs | **86.91%** |

Spawn head: balanced accuracy **99.93%** — no never-spawn collapse.

**BC eval vs V71 (20 deterministic games)**: **10/20 = 50.0%**, mean halite
**12,445 vs 11,675**. Parity with the teacher from imitation alone — about the
best BC can do, and the green light for PPO. Notable variance: blowout wins
(57.0k vs 4.2k) but also collapse games (497 halite in one game, two more weak
ones) — classic off-distribution snowballing. Those collapse games are PPO's
cheapest win-rate headroom: it doesn't need to outplay V71 there, just not
fall apart.

**PPO run COMPLETE** (`rl_train.py --episodes 300 --device cuda`, ≈70-100s/ep).
**PPO improved on BC — the first time across v6/v7/v8/v9 that RL made the
imitation policy better instead of worse.**

Gate-eval trajectory (every 20 eps, 9 deterministic games each):

| Episode | Win rate | Margin | Note |
|---|---|---|---|
| 0 (BC baseline) | 0.78 | −1712 | initial `best.pt` |
| 20 | 0.44 | −6299 | |
| 40 | 0.67 | −2934 | |
| 60 | 0.44 | −6055 | |
| **80** | **0.78** | **+1816** | *** new best *** |
| 100 | 0.67 | −1848 | |
| 120 | 0.56 | −6506 | |
| 140 | 0.67 | +584 | |
| **160** | **0.78** | **+5973** | *** new best — final champion *** |
| 180 | 0.67 | −2090 | |
| 200 | 0.44 | −7612 | |
| 220 | 0.44 | −3028 | |
| 240 | 0.56 | −2170 | |
| 260 | 0.56 | +834 | |
| 280 | 0.67 | +3526 | |
| 300 (two seed sets) | 0.56 / 0.67 | −4462 / +992 | endpoint weaker than the ep-160 peak |

* `best.pt` = the **episode-160** policy: same 7/9 wr as BC but margin swung
  -1712 → +5973 (≈7.7k) — PPO fixed the blowout-loss pattern rather than
  winning more eval games. Eval mean deposits 10.4k (BC) → 15-18k late.
* What PPO changed, early vs late run (rollout batches of 6 games):

  | Rollout stat | Early (eps 16-60) | Late (eps 200-300) |
  |---|---|---|
  | Kills per batch | 1-8 | 8-23 (learned collision-trading — no scripted enemy avoidance) |
  | Spawns per batch | ≈120-190 | ≈200-280 |
  | Deposit batches | ≈9-21k | ≈11-32k (peak 31.7k, ep 201) |
  | Entropy | 0.12-0.18 | 0.14-0.22 — slight RISE, still exploring, no collapse |
  | KL per update | 0.001-0.006 | 0.003-0.013 — never hard-tripped the 0.02 brake |
  | vloss | spiked ≈2.0 at ep 16 | settled ≈0.5-1.2 |
* Post-peak wandering: evals after ep 160 oscillated 0.44-0.67 and the ep-300
  endpoint (0.56/0.67) is weaker than the ep-160 snapshot — the gate, not the
  endpoint, is what saved the peak (rl_v8 would have shipped the endpoint).
* Note: the log prints TWO `[eval ep 300]` lines — the scheduled in-loop eval
  plus a final post-loop `run_eval_and_gate` on different seeds (also saves
  `ppo_final.pt`).

### FINAL VERDICT (2026-07-03): GOAL MET — 258/450 = 57.3% vs V71

Five eval batches of `best.pt` (deterministic, random seeds):

| Batch | Games | Wins | Win rate | Mean halite (rl_v9 / V71) |
|---|---|---|---|---|
| 1 | 50 | 34 | 68.0% | 15,933 / 14,411 |
| 2 | 100 | 59 | 59.0% | 16,016 / 15,364 |
| 3 | 100 | 52 | 52.0% | 15,772 / 17,300 |
| 4 | 100 | 54 | 54.0% | 15,625 / 15,522 |
| 5 | 100 | 59 | 59.0% | 15,911 / 16,452 |
| **Pooled** | **450** | **258** | **57.3%** | **15,842 / 15,965** |

**95% CI 52.8-61.9% — the whole interval clears 50%** (p ≈ 0.001). A bot
with fully learned moves, spawning, dropoffs, and enemy-collision behaviour
beats the strongest 2019 hand-coded bot. The early "decline" (68→59→52)
reversed in batches 4-5 — pure sampling scatter around a frozen model; ≈57%
is the definitive strength estimate.

Pooled mean halite is a caveat (see table): V71 slightly ahead despite
losing 57% of games. rl_v9 wins more often; V71 wins bigger (blowouts up to
74,177 in batch 5) when its economy runs free. rl_v9's edge is suppression +
consistency, not raw mining — "out-mine V71 on a free board" is the natural
rl_v10 objective.

Outcome texture (consistent across all batches):

| Pattern | Evidence |
|---|---|
| Suppressive wins | V71 repeatedly held under 1k (141, 148, 268, 289, 321, 367, 587, 592, 755, 876, 939...) |
| Suppression is positional, not attrition | Replay scan of the 50-game batch: 35 enemy collisions total (18 games had any), all clean 1-for-1 trades (35 lost / 35 killed) — the freeze comes from squatting V71's target halite and triggering its hand-coded enemy-avoidance |
| Disciplined fleet | 0 open-field self-collisions; 283 own-structure endgame bankings (intentional) |
| Bimodal losses | V71's wins are blowouts (40-74k); rl_v9 rarely loses close — whichever bot's game plan takes hold wins |
| Coin-flip margin games | Won by 23 halite in one game; lost by 7 in another |
| Small samples lie | 5-game spot checks scored 1/5 and 4/5 on the same model — judge only on ≥50-game evals |

Replays of the 50-game batch plus two 5-game viewing batches were saved to
`rl_v9/replays/` via `--save-replays` (local only — `*.hlt` is gitignored;
regenerate with the command above).

### Head-to-head vs rl_v8 (2026-07-03): 75/100 = 75.0%

Direct pitch of `rl_v9/checkpoints/best.pt` against rl_v8's strongest
checkpoint `rl_v8/checkpoints/bc.pt` (rl_v8's PPO degraded its BC policy,
so BC is its best). Both deterministic, via `run_game.py` on 32x32, seats
alternated each game, no replays kept.

| Batch | Seeds | Games | rl_v9 wins | Win rate | Mean halite (rl_v9 / rl_v8) |
|---|---|---|---|---|---|
| Pilot | 1-10 | 10 | 6 | 60.0% | 23,695 / 16,828 |
| Main | 11-110 | 100 | 75 | 75.0% | 29,720 / 16,674 |
| **Pooled** | 1-110 | **110** | **81** | **73.6%** | — |

Main batch: 95% CI 66.5-83.5%, binomial p < 0.0001 vs 50% — conclusive.
Unlike the V71 matchup (57.3% wins but out-mined on average), rl_v9
dominates rl_v8 on both axes, out-mining it ≈1.8x. The 6/10 pilot
understated the gap — small samples lie in this matchup too.

## Reading the rl_train log (notes from the 2026-07-03 run)

All games — evals and rollouts — are rl_v9 vs the real V71 bot
(`--opponent` defaults to `v71`); winner = most halite deposited.

* **Episode line fields**: `dep=ours/theirs` and `wr` are from that episode's
  **6 stochastic rollout games** (sampled actions = exploration noise, so
  wr swinging 0.5-1.0 between episodes is scatter, not learning).
  `kl`/`ent` are exactly 0 during `[warmup]` — proof the policy is untouched;
  they go nonzero when the tag flips to `[ppo]` at ep 16 (watch kl stay
  ≤ ≈0.02, the early-stop target). `lbc` = BC-anchor KL coefficient,
  annealing 1.0 → 0.1 over 150 eps. `drop`/`spawn` = learned heads acting.
* **`vloss`** = critic (value net) MSE vs realized returns. The only live
  training signal during warmup. It is *bouncy by nature* — each episode is
  scored against 6 fresh games whose outcomes vary hugely — so judge the
  level, not monotonicity. Warmup exists so this drops before any policy
  gradient flows (rl_v8 failure #2).
* **`wrecks` vs `kills`**: wrecks = our ships destroyed, kills = enemy ships
  destroyed (a head-on trade increments both). High wrecks with near-zero
  kills is NOT alarming: the endgame recall piles the fleet onto own
  structures, which counts as wrecks but banks the cargo (credited as
  deposit). Watch kills/outcomes for whether collision-trading is learned.
* **Small-sample warning**: evals are only 9 deterministic games
  (`--eval-games 9`). BC's 0.78 baseline = 7/9 — consistent with the same
  ≈50-65% bot that scored 10/20 in the standalone eval. Negative margin at
  high wr = narrow wins, blowout losses (the BC collapse-game pattern).
* **Gate consequence**: a lucky-high baseline sets a *stricter* gate — PPO
  needs 8/9, or 7/9 with better margin, to log `*** new best ***`. Slower to
  show progress but safe; if no new best ever appears, `best.pt` still holds
  BC (the "PPO can't silently regress" guarantee). Don't panic before
  ep 40-60: lbc keeps the policy leashed to BC early.
