# Revised Implementation Plan: REMC Acceptance Rate Fix

## Summary of the Problem

From the last run's `frame_scores.csv`, we observe:
- Scores: `4493.6 → 4491.2 → 4487.7` over 20 frames
- Acceptance rate: **1.0%** (only 2 moves accepted out of ~200 proposals)
- Temperature: constant `1.0` across all frames

The IMP Metropolis criterion is: $P(\text{accept}) = e^{-\Delta E / T}$

With $T = 1.0$, a proposed move that worsens the score by just $\Delta E = +10$ yields $P = e^{-10} \approx 4.5 \times 10^{-5}$. The rigid body mover (default 4Å translation, 0.04 rad rotation) regularly produces score deltas of $+50$ to $+500$, making virtually all uphill moves impossible.

---

## Recommendation: **Per-Point Score Normalization via PMI Weight**

### Why This Approach

I recommend **Option A (Score Scaling)** using PMI's built-in `set_weight()` mechanism, specifically normalizing the score by the number of data points in the cluster.

| Criterion | Score Scaling (weight=1/N) | Temperature Scaling (T=50000) |
|---|---|---|
| **Physical meaning** | Per-localization log-likelihood — natural, interpretable | Arbitrary temperature with no physical basis |
| **Cluster-size independence** | ✅ Same config works for all clusters | ❌ Need different T for different N |
| **IMP compatibility** | ✅ Uses built-in `set_weight()` on `RestraintBase` | ⚠️ Works, but PMI's temp ladder was designed for T≈1-10 |
| **Thesis clarity** | Easy to explain: "normalized log-likelihood" | Awkward: "we inflate temperature by 10,000×" |
| **Scoring comparisons** | ✅ Scores are per-point, directly comparable | ❌ Raw scores mixed with inflated temps |

### Concrete Numbers

For the most recent run (cluster 751, N=893, GMM scoring):
- Raw total score: `4493.6`
- Per-point score: `4493.6 / 893 ≈ 5.03`
- Weight: `1/893 ≈ 0.00112`

With per-point normalization and $T = 1.0$:
- A move worsening score by $\Delta E_{raw} = +100$ → $\Delta E_{weighted} = +0.112$
- $P = e^{-0.112} = 0.894$ → **89% acceptance** (too high, but tunable)

With $T = 0.5$ (slightly colder):
- $P = e^{-0.224} = 0.799$ → **80% acceptance** (good exploration)

> [!TIP]
> The optimal acceptance rate for REMC is **20-40%** (literature consensus for structural sampling). We should target this range by tuning the weight/temperature combination after the initial implementation.

---

## Proposed Changes

### [MODIFY] [scoring_restraint.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/restraint/scoring_restraint.py)

No code changes needed in the restraint itself. The `ScoringRestraintWrapper` already inherits `set_weight()` from `IMP.pmi.restraints.RestraintBase`. The weight is applied automatically via `self.weight * self.rs.unprotected_evaluate(None)`.

### [MODIFY] [mcmc_sampler.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/simulation/mcmc_sampler.py)

Add a `score_weight` parameter to `run_bayesian_sampling()`. Before calling `add_to_model()`, call `scoring_restraint_wrapper.set_weight(score_weight)`.

### [MODIFY] [NPC_example_BD.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/NPC_example_BD.py)

Pass `score_weight=1.0/n_cluster_points` when invoking the Bayesian sampler.

### [MODIFY] [pipeline_config.json](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/pipeline_config.json)

Add optional `score_weight` to the `bayesian` config block. When set to `"auto"`, compute `1/N` automatically. When set to a float, use that value directly.

```json
"bayesian": {
    "run_sampling": true,
    "scoring_type": "GMM",
    "number_of_frames": 200,
    "monte_carlo_steps": 50,
    "score_weight": "auto"
}
```

---

## Secondary Adjustments

### Increase Sampling Volume

Now that acceptance will be reasonable, we should also increase sampling depth:
- `number_of_frames`: `20 → 200` (at minimum; 500-1000 for production)
- `monte_carlo_steps`: `10 → 50` (more proposals per frame)

### Temperature Ladder

Keep the default PMI temperature range but widen slightly:
- `replica_exchange_minimum_temperature`: `1.0` (unchanged)
- `replica_exchange_maximum_temperature`: `2.5 → 5.0` (wider exploration at high T)

---

## Open Questions

> [!IMPORTANT]
> 1. Do you agree with using per-point normalization (`weight = 1/N`) as the default `"auto"` behavior?
> 2. Should we also expose the `max_rb_trans` (rigid body translation step size, default 4Å) and `max_rb_rot` (rotation step, default 0.04 rad) in the config? These directly control how "big" each proposed move is, which also affects acceptance rate.

## Verification Plan

### Automated Tests
1. Run the pipeline with `score_weight="auto"` and `number_of_frames=100`.
2. Verify acceptance rate is between **15-50%** in `frame_scores.csv`.
3. Verify that the score trajectory shows clear exploration (not monotonically flat).
4. Verify that the `av_trajectory.rmf3` shows visible AV movement across frames in ChimeraX.

### Manual Verification
- Compare the final AV coordinates between the old run (1% acceptance) and the new run to confirm meaningful structural exploration has occurred.
