# Per-Step Coordinate Logging for All Optimization Modes

## Goal
Add temporary per-step coordinate printing to all three optimization modes (Frequentist, Bayesian/REMC, Brownian) so we can verify that the IMP optimizers are actually moving the model particles. Once verified, the logging will be removed.

## Proposed Changes

---

### Frequentist — [frequentist_optimizer.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/simulation/frequentist_optimizer.py)

**Current**: Single call `cg.optimize(max_cg_steps)` (line 78) — no visibility into intermediate states.

**Change**: Replace with a loop of `cg.optimize(1)` repeated `max_cg_steps` times. After each step, print the current score and AV coordinates.

```python
# Replace: cg.optimize(max_cg_steps)
for step in range(max_cg_steps):
    cg.optimize(1)
    coords = [IMP.core.XYZ(av).get_coordinates() for av in avs]
    score = sf.evaluate(False)
    print(f"  [CG Step {step+1}/{max_cg_steps}] Score: {score:.4f}  Coords: {[list(c) for c in coords]}")
```

> [!WARNING]
> Running CG one step at a time may converge differently than a single `optimize(N)` call, because IMP's CG resets its line-search state on each `optimize()` invocation. This is fine for debugging but the single-call version is better for production.

---

### Bayesian/REMC — [mcmc_sampler.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/simulation/mcmc_sampler.py)

**Current**: `rex.execute_macro()` runs in a background thread (line 100). The main thread monitors `stat.0.out` for progress but never reads coordinates.

**Change**: In the monitoring loop (lines 112-121), after detecting a new frame, read the live AV coordinates from the IMP model and print them. Since IMP updates coordinates in-place on the particles, we can read them from the main thread between frames.

```python
# Inside the monitoring while-loop, after detecting new frames:
if new_frames_count > current_frame:
    coords = [list(IMP.core.XYZ(av).get_coordinates()) for av in avs]
    score_val = scoring_restraint_wrapper.evaluate()
    print(f"  [REMC Frame {new_frames_count}] Score: {score_val:.4f}  Coords: {coords}")
```

> [!NOTE]
> The coordinates read between frames are approximate snapshots — the REMC thread may be mid-update. This is acceptable for debugging but not for production analysis.

---

### Brownian Dynamics — [simulation_setup.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/brownian_dynamics/simulation_setup.py)

**Current**: Single call `bd.optimize(number_of_bd_steps)` (line 157).

**Change**: Replace with a loop of `bd.optimize(1)` repeated `number_of_bd_steps` times, printing coordinates and score at each step.

```python
# Replace: bd.optimize(number_of_bd_steps)
for step in range(number_of_bd_steps):
    bd.optimize(1)
    coords = [list(IMP.core.XYZ(av).get_coordinates()) for av in avs]
    score = scoring_function.evaluate(False)
    print(f"  [BD Step {step+1}/{number_of_bd_steps}] Score: {score:.4f}  Coords: {coords}")
```

---

## Verification Plan

### Test Sequence
Run the pipeline three times, once per mode, using `pipeline_config.json`:

1. **Frequentist**: Set `optimization.mode` to `"frequentist"`, run, verify coordinates change each CG step.
2. **Bayesian**: Set `optimization.mode` to `"bayesian"`, run, verify coordinates change each REMC frame.
3. **Brownian**: Set `optimization.mode` to `"brownian"`, run, verify coordinates change each BD step.

### Success Criteria
- Coordinates printed at each step are **not all identical** (proves the optimizer is moving particles).
- Score values change over time (proves the scoring function responds to coordinate changes).
- No crashes or NaN values.

### Cleanup
Once all three modes are verified, **revert all print-statement changes** to restore production behavior.
