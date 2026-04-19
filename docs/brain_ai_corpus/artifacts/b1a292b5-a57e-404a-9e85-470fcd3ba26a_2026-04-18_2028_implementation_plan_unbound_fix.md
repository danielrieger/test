# Pipeline Stabilization: Final Plan

## Background & Git History Analysis

I traced the evolution of the Bayesian score weight across all 4 commits in the repository:

### Commit 0 (`0121c2b` — "docs: README, examples")
- **No `score_weight` in config.** The `run_bayesian_sampling` call passed no weight argument.
- The function default was `score_weight: float = 1.0`.
- **Effective weight: always `1.0`.**
- Config: `number_of_frames: 20`, `monte_carlo_steps: 10` (very short test runs).
- Filtering: `"random"` at 15%.

### Commit 1 (`1cdb892` — "feat: optimize Bayesian REMC sampling")
- **`"score_weight": "auto"` added to config.**
- New logic added: `effective_weight = 1.0 / n_points if BAYESIAN_SCORE_WEIGHT == "auto" else float(BAYESIAN_SCORE_WEIGHT)`
- `n_points` was the cluster's localization count, available directly in the loop scope.
- Config: `number_of_frames: 200`, `monte_carlo_steps: 50` (production runs).
- Filtering: `"random"` at 15%.

### Current state (post-refactoring, not committed)
- Weight calculation moved to `trigger_opt()` function.
- **Bug introduced**: `n_pts = sr.dataxyz.shape[0] if hasattr(sr, 'dataxyz') else 1`
- `ScoringRestraintWrapper` never exposes `dataxyz` → `hasattr` always `False` → weight always `1.0`.

---

## Proposed Changes

### Fix 1: Pass `n_pts` explicitly to `trigger_opt` (Bug fix)

#### [MODIFY] [NPC_example_BD.py](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\examples\NPC_example_BD.py)

**Why**: The current code tries to read `n_pts` from the restraint wrapper object, which doesn't expose it.

**What**:
1. Add `n_pts` to `trigger_opt`'s function signature.
2. Replace the faulty `hasattr` check with the passed `n_pts`.
3. Update the call site in `run_evaluation` to pass `n_pts`.
4. Add diagnostic print: `print(f"    Effective weight: {w:.6f} (n_pts={n_pts})")`.

---

### Fix 2: Weight capping fallback for full-map mode

#### [MODIFY] [NPC_example_BD.py](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\examples\NPC_example_BD.py)

**Why**: The `"auto"` weight formula `1/n_points` was designed for `"random"` filtering (~175 pts → weight ~0.006 → ~3.6 kT). On the full map (~1172 pts → weight ~0.0009 → ~0.5 kT), the energy contribution becomes too weak to guide the sampler. Users should be able to use `"none"` filtering without the sampling becoming meaningless.

**What**: When `score_weight == "auto"`, apply a floor:
```python
w = max(1.0 / n_pts, 0.005)  # Ensures energy ≥ ~3 kT for typical GMM scores
```

This preserves the original behavior for small clusters (random filtering) while preventing the weight from becoming negligibly small on large clusters (full map).

---

### Fix 3: Runtime config validation (Prevention)

#### [MODIFY] [NPC_example_BD.py](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\examples\NPC_example_BD.py)

**Why**: The config mismatch (changing `"random"` to `"none"` without adjusting the weight) happened silently. There should be guardrails that warn users when they create potentially problematic combinations.

**What**: Add a `validate_config(config)` function called after `load_config()` that checks for:

1. **Filtering + weight interaction**:
   ```
   WARNING: filtering.type="none" with score_weight="auto" may produce 
   very weak sampling constraints on large clusters. Consider using 
   "random" filtering or setting a manual score_weight value.
   ```

2. **Scoring type + optimization mode incompatibility** (already partially restored):
   ```
   WARNING: frequentist/brownian optimization does not support GMM scoring.
   ```

3. **EMAN2 + "none" filtering → no noise clusters**:
   ```
   INFO: With eman2 clustering on unfiltered data, noise separation 
   tests will be skipped (all clusters are valid NPCs).
   ```

These are `print()` warnings, not exceptions — they inform but don't block execution.

---

### Fix 4: Self-documenting config template

#### [NEW] [pipeline_config_template.jsonc](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\examples\pipeline_config_template.jsonc)

**Why**: The user requested a config file where all available options are visible with explanations. JSON doesn't support comments natively, so we'll create a `.jsonc` (JSON with Comments) reference template that documents every field. The actual `pipeline_config.json` remains a clean, parseable JSON file.

**What**: Create a template like:
```jsonc
{
    // === DATA PATHS ===
    "paths": {
        "smlm_data": "ShareLoc_Data/data.csv",     // Path to SMLM localization CSV
        "pdb_data": "PDB_Data/7N85-assembly1.cif",  // PDB or MMCIF structure file
        "av_parameters": "av_parameter.json",        // AV labeling parameters
        "downsample_residues_per_bead": 10           // Coarse-graining resolution
    },

    // === EXECUTION ===
    "execution": {
        "test_scoring_types": ["Tree", "GMM", "Distance"],  // Options: "Tree", "GMM", "Distance"
        "target_cluster_id": null  // null = auto-select largest | "random" = random NPC | integer = specific ID
    },

    // === FILTERING ===
    // Controls which portion of the SMLM data is processed.
    // NOTE: "auto" score_weight was designed for "random" filtering.
    //       Using "none" with "auto" weight will cap the weight at 0.005.
    "filtering": {
        "type": "random",  // Options: "none" (full map), "random" (spatial subset), "filter" (manual cut)
        "filter": { ... },
        "random": {
            "size_percentage": 15  // Percentage of data to include (1-100)
        }
    },

    // === OPTIMIZATION ===
    "optimization": {
        "mode": "bayesian",  // Options: "bayesian", "frequentist", "brownian"
        "bayesian": {
            "run_sampling": true,
            "scoring_type": "GMM",        // Which restraint to optimize. Options: "Tree", "GMM", "Distance"
            "number_of_frames": 200,      // REMC frames (more = better sampling, slower)
            "monte_carlo_steps": 50,      // MC steps per frame
            "score_weight": "auto",       // "auto" = 1/n_points (capped at min 0.005) | float = manual weight
            "max_rb_trans": 4.0,          // Max rigid body translation per step (Angstroms)
            "max_rb_rot": 0.04            // Max rigid body rotation per step (radians)
        },
        // ... frequentist, brownian sections with similar comments
    }
}
```

Additionally, update `load_config()` to print a note pointing users to this template if they encounter warnings.

---

### Fix 5: Demote Separation test to informational metric

#### [MODIFY] [validation.py](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\src\smlm_score\validation\validation.py)

**Why**: The Separation test compares valid NPC scores against noise cluster scores. With EMAN2 on high-quality data, noise clusters simply don't exist. This is a sign of good data quality, not a pipeline failure. Counting it as a "FAIL" in the report is misleading.

**What**:
1. When noise clusters are available → run the test and report results as before (PASS/FAIL).
2. When no noise clusters exist → print an informational message instead of a failure:
   ```
   [~ SKIP] Separation_Tree: No noise clusters available (all clusters are valid NPCs). 
            This is expected with high-quality EMAN2 particle picking.
   ```
3. **Do not count skipped tests toward the pass/fail total.** If 4 tests run and 2 are skipped, the report should say `Total: 2/4 passed` not `Total: 2/6 passed`.

> [!NOTE]
> The Separation test itself remains in the codebase. It will naturally activate whenever noise clusters are present (e.g., when using HDBSCAN clustering or `"random"` filtering that clips EMAN2 boxes). We are not deleting it — just making it gracefully optional.

---

### Fix 6: Revert config default to `"random"`

#### [MODIFY] [pipeline_config.json](file:///\\wsl.localhost\Ubuntu\home\daniel\Thesis\smlm_score\examples\pipeline_config.json)

**Why**: `"random"` is the tested, validated default. It produces appropriately-sized clusters for the `"auto"` weight and naturally generates noise candidates for the Separation test.

**What**: Change `"type": "none"` back to `"type": "random"`.

---

## Summary

| # | Change | File | Why |
|---|--------|------|-----|
| 1 | Fix weight bug (pass `n_pts`) | `NPC_example_BD.py` | `hasattr` always fails on wrapper |
| 2 | Cap weight at `max(1/n, 0.005)` | `NPC_example_BD.py` | Prevents meaningless weights on full map |
| 3 | Add `validate_config()` warnings | `NPC_example_BD.py` | Prevents silent misconfigurations |
| 4 | Create documented config template | `pipeline_config_template.jsonc` | Shows all options with explanations |
| 5 | Demote Separation to optional | `validation.py` | Graceful skip when no noise exists |
| 6 | Revert config to `"random"` | `pipeline_config.json` | Restore tested default |

## Verification Plan

1. Run with `"random"` filtering → verify weight prints ~0.006, REMC converges, Separation tests find noise clusters.
2. Run with `"none"` filtering → verify weight is capped at 0.005, warning is printed, Separation tests show `[~ SKIP]` instead of `[x FAIL]`.
3. Deliberately set an incompatible config (e.g., `frequentist` + `GMM`) → verify warning is printed at startup.
4. Verify `pipeline_config_template.jsonc` contains accurate documentation for all fields.
