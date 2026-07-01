# rl_v6 — pure-RL imitation (and improvement) of rl_v5

> **STATUS: ABANDONED — 0% win rate at all stages. See [Conclusions](#conclusions) before reading further.**

rl_v6 is a research vehicle that asks: can a **purely learned** neural policy
reproduce — and ideally beat — rl_v5, whose strength is mostly *rule-based*?

rl_v5 wins by a hand-coded FSM (PROSPECT→HARVEST→HOME→ESCAPE) fed to its net as
features **and** a logit prior (`FSM_PRIOR_LOGIT=3.0`, so the FSM is ~71 % of the
policy and always the greedy choice), then post-processed by homing/prospect
resolution, 4-phase collision avoidance, an endgame rule and a spawn economy gate.
The network barely matters. rl_v6 throws **all** of that away:

> per ship → pure features (14ch + 29 base scalars) → `ActorCritic(29, 6)` →
> primitive move (STAY/N/E/S/W/DROPOFF), executed **verbatim**.
> per turn → global features → `SpawnHead` → spawn yes/no.

No FSM, no prior, no homing/collision/endgame/spawn rules at inference (the only
non-net constraint is the engine-legality mask on DROPOFF). Everything rl_v6 does
must be *learned* from how rl_v5 behaves. This is faithful but means an untrained
rl_v6 self-collides and loses ships — that is the cost of 100 % purity, recovered
through DAgger and PPO.

rl_v1–v5 and `my_extension/halite_engine.py` are left intact; rl_v5/rl_v4 are
imported read-only as the imitation expert and as opponents.

## Quick start (one command)

`run_pipeline.py` does the whole thing — generate → extract → BC → DAgger cycles
→ (optional PPO) — and after every milestone it benchmarks rl_v6 vs rl_v5/rl_v4,
appends to `checkpoints/progress.csv`, and reprints an improvement table so you can
watch it get better.

```bash
# imitate rl_v5 (BC + 4 DAgger cycles); fastest path to a working bot
python run_pipeline.py --games 80 --selfplay-games 20 --dagger-cycles 4

# also try to BEAT rl_v5 with PPO afterwards (slow — hours)
python run_pipeline.py --games 80 --dagger-cycles 4 --ppo-episodes 2000

# reuse data already generated; just keep improving
python run_pipeline.py --skip-gen --dagger-cycles 4
```
Progress table columns: `bc_match` (held-out mimicry of rl_v5), and per opponent
`win%`, mean rl_v6 halite, mean opponent halite. You want `bc_match` and the
`win%` rising across the BC → DAgger1 → DAgger2 … rows.

The individual scripts below are what the orchestrator calls; use them directly
only when debugging a single stage.

## Pipeline stages (what run_pipeline.py runs for you)

Run everything from `my_extension/rl_v6/`. On Windows set `PYTHONIOENCODING=utf-8`.

### 1. Generate rl_v5 gameplay
```bash
python gen_data.py --n-games 100 --mode both --out-dir replays_v6/
```
`vs_v4` (rl_v5 vs rl_v4) and `selfplay` (rl_v5 vs rl_v5) games; rl_v5 plays
stochastically by default for state diversity (`--deterministic` for the
canonical policy).

### 2. Extract the imitation dataset
```bash
python rl_collect.py replays_v6/ --output dataset/ --both
```
Emits, per replay/player: pure `(obs, action 0-5)` ship-steps (construct labelled
DROPOFF) **and** a per-turn spawn dataset (global features + 0/1 label).

### 3. Behavioral cloning
```bash
python bc_train.py --data dataset/ --epochs 15 \
    --out checkpoints/model_weights.pt --spawn-out checkpoints/spawn_weights.pt
```
Class-weighted cross-entropy (STAY dominates; DROPOFF is rare) + a BCE spawn head.
Reports held-out per-action match accuracy (mimicry quality).

### 4. DAgger cycles (fix distribution shift)
```bash
# collect on-policy states labelled by the rl_v5 expert, then retrain
python dagger.py --policy checkpoints/model_weights.pt \
    --spawn checkpoints/spawn_weights.pt --n-games 20 --opponent rl_v5 \
    --out dataset/ --retrain --epochs 10
```
rl_v6 drives player 0 (on-policy); the rl_v5 expert (`experts.FrozenBotDriver`,
rl_v5 run in module isolation) labels every visited state. New shards land in
`dataset/`, so retraining uses BC + DAgger data together. Repeat several cycles.

### 5. PPO fine-tune (try to exceed rl_v5)
```bash
python rl_train.py --resume checkpoints/model_weights.pt \
    --spawn checkpoints/spawn_weights.pt --opponent rl_v5 \
    --episodes 4000 --checkpoint-dir checkpoints/
```
Warm-starts from BC/DAgger, refines the pure movement policy. The spawn head is
loaded **frozen** (focuses the gradient on movement). Reward is rl_v5's
deposited-anchored design with the `reward_scale`/entropy collapse guards. Watch
`checkpoints/training_log.csv`: `deposited` rising, `mean_entropy` not → 0, a
non-uniform action distribution.

### Benchmark
```bash
python rl_eval.py --opponent rl_v5 --games 30 --deterministic
python rl_eval.py --opponent rl_v4 --games 30 --deterministic
```

## Files
| file | role |
|------|------|
| `rl_config.py` | pure action space (6) + global spawn feature builder |
| `rl_model.py` | `ActorCritic` (copied from rl_v5; used as `(29, 6)`) |
| `spawn_model.py` | `SpawnHead` — learned spawn decision |
| `rl_bot.py` | **pure** inference bot (no rules) |
| `rl_features.py` | feature extractors (copied; only the base block is used) |
| `gen_data.py` / `rl_collect.py` | replay generation / dataset extraction |
| `bc_train.py` | behavioral cloning |
| `experts.py` | `FrozenBotDriver` — rl_v5/rl_v4 in isolation (DAgger expert + PPO opponent) |
| `dagger.py` | DAgger collection (+ optional retrain) |
| `rl_env.py` | pure player-0 environment (`HaliteEnvV6`) |
| `rl_train.py` | PPO fine-tuning |
| `rl_eval.py` | head-to-head benchmark |
| `run_pipeline.py` | **one-command orchestrator** + progress tracking |

## Notes
- Every game runs on the Python `halite_engine.py` (no C++).
- The expert/opponent loader (`experts.FrozenBotDriver`) imports rl_v5/rl_v4
  modules in isolation (clears `rl_features`/`rl_model`/`rl_env` from the import
  cache), so each frozen bot uses its OWN code and rules.
- Trained weights are gitignored by default — commit deliberately to share a run.

---

## Conclusions

**rl_v6 was abandoned after 4 DAgger cycles. It achieved 0% win rate against rl_v5 at every stage.**

### Actual results (from `checkpoints/progress.csv`)

| Stage | BC match | Win% vs rl_v5 | rl_v6 deposits | rl_v5 deposits |
|-------|----------|---------------|----------------|----------------|
| BC | 51.7% | 0% | 1,355 | 14,463 |
| DAgger 1 | 60.7% | 0% | 548 | 7,225 |
| DAgger 2 | 68.3% | 0% | 300 | 9,715 |
| DAgger 3 | 75.5% | 0% | 708 | 10,438 |
| DAgger 4 | 80.7% | 0% | 555 | 8,891 |

**The DAgger paradox**: match rate improved from 51.7% → 80.7% (the model reproduced rl_v5's moves with increasing fidelity), yet deposits simultaneously *decreased* from 1,355 → 555. Higher mimicry accuracy = worse actual game performance.

### Why it failed

**1. Economy collapse without explicit rules.**
rl_v5 guards its economy with hardcoded rules: spawn only when `bank >= 1000`, reserve funds for dropoffs, abort spawn if the factory is occupied. rl_v6 replaces all of this with a learned `SpawnHead`. In practice the spawn head was too conservative — rl_v6 ran tiny fleets (1–3 ships) while rl_v5 ran 8–12. With so few ships, even perfect movement produces negligible halite.

**2. Self-collisions without a collision resolver.**
rl_v5 runs a 4-phase deterministic collision resolver after the NN outputs actions. rl_v6 executes primitive moves verbatim. The learned policy inevitably self-collides, destroying loaded ships and erasing the halite they were carrying. This is an unrecoverable compounding loss.

**3. Same FSM-observability ceiling as rl_v7 (but without any safety net).**
Just like rl_v7's 58.5% BC ceiling (see `rl_v7/README.md`), rl_v6 cannot predict rl_v5's FSM state (HARVEST/HOME/ESCAPE) from a single-frame feature vector. At 80.7% match rate, rl_v6 is still wrong 1 turn in 5 — and without a safety resolver, each wrong move is directly destructive.

**4. DAgger made the economy problem worse.**
DAgger collects on-policy states (rl_v6 driving, rl_v5 labelling). As rl_v6's economy collapsed into small fleets, the states it visited became unrepresentative of rich-fleet play. Retraining on those impoverished states made rl_v6 better at mimicking rl_v5's behaviour *in impoverished states*, not in normal play.

### Key lesson for future bots

> **You cannot strip out rl_v5's structural rules and expect a neural network to re-learn them from observations alone.**

rl_v5's strength is its rules, not its neural network. The network barely contributes — the FSM prior (`FSM_PRIOR_LOGIT = 3.0`) means the FSM is ~71% of every action. Attempting to imitate only the surface behaviour (action labels) while discarding the rules leads to compounding failures across collision, economy, and state coverage.

The successor project **rl_v7** (`../rl_v7/`) addressed this by keeping the structural safeguards (deterministic resolver, copied spawn/dropoff economy) and learning only ship strategy on top of them. Even with those fixes, the FSM-observability ceiling (~58.5% BC match rate) prevented rl_v7 from beating rl_v5. The remaining gap requires explicitly adding FSM state as input features.
