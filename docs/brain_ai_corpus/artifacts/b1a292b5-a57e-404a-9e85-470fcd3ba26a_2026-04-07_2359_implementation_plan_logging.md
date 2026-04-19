# Implementation Plan — Sampling Trajectory Logger

## Goal

Export a CSV file containing the XYZ coordinates of all model particles (AVs), the scoring value, and frame metadata at **every sampling frame** during Bayesian REMC. This will allow the user to verify that (a) coordinates are actually changing between frames, and (b) the score evolves as expected.

## Background & Design Rationale

The IMP.pmi `ReplicaExchange` macro calls `get_output()` on every object in the `output_objects` list **once per frame**. The returned dict is serialized into `stat.0.out`. This is the only reliable hook we have — the macro controls the MC loop internally and does not expose per-step callbacks.

> [!IMPORTANT]
> **Key insight**: Rather than creating a new output object class (which risks IMP.pmi compatibility issues), we will **instrument the existing `ScoringRestraintWrapper.get_output()`** to additionally write trajectory data to a side-channel CSV file. This guarantees perfect synchronization between the stat file and our trajectory log.

## Output File Format

**File**: `<output_dir>/trajectory_trace.csv`

Each row = one sampling frame. Columns:

| Column | Description |
|--------|-------------|
| `frame` | Frame index (0-based) |
| `score` | Raw scoring value at this frame |
| `score_objective` | Negated score (IMP objective value) |
| `av_0_x`, `av_0_y`, `av_0_z` | XYZ of particle 0 (in nm, data-space) |
| `av_1_x`, `av_1_y`, `av_1_z` | XYZ of particle 1 |
| ... | ... for all 32 AV particles |
| `centroid_x`, `centroid_y`, `centroid_z` | Mean position of all AVs |
| `rmsd_from_initial` | RMSD relative to the starting configuration |

> [!TIP]
> The centroid and RMSD columns provide instant diagnostics: if the centroid drifts far or RMSD stays at zero, sampling is broken.

## Proposed Changes

### 1. [MODIFY] `src/imp_modeling/restraint/scoring_restraint.py`

Add trajectory logging capability to `ScoringRestraintWrapper`:

- Add `enable_trajectory_logging(output_dir)` method that:
  - Creates a CSV writer for `trajectory_trace.csv`
  - Stores the initial model coordinates for RMSD calculation
  - Sets `self._trajectory_enabled = True`
- Modify `get_output()` to call `self._log_trajectory_frame()` when logging is enabled
- `_log_trajectory_frame()`:
  - Reads current model coords via `self.scoring_restraint_instance._current_model_coords()`
  - Computes centroid, RMSD from initial
  - Appends one row to the CSV writer

### 2. [MODIFY] `src/imp_modeling/simulation/mcmc_sampler.py`

In `run_bayesian_sampling()`, after constructing the `ScoringRestraintWrapper` reference:

```python
# Enable trajectory logging before starting REMC
scoring_restraint_wrapper.enable_trajectory_logging(output_dir)
```

No other changes needed — the logging hooks into the existing `get_output()` call chain.

### 3. No new files required

Everything is contained within the existing restraint and sampler modules.

---

## Open Questions

> [!WARNING]
> **Coordinate space**: The logged coordinates will be in the **data-aligned nm space** (after scaling + offset + PCA rotation). This is the "working" coordinate system. Is this the space you want, or do you need the raw IMP Angstrom-space coordinates as well?

## Verification Plan

### Automated Tests
1. Run the pipeline with `number_of_frames=5` and `monte_carlo_steps=10`
2. Verify `trajectory_trace.csv` exists and has exactly 5 data rows
3. Check that coordinate columns contain non-identical values across rows (proving movement)
4. Verify RMSD column is zero for frame 0 and non-zero for subsequent frames

### Manual Verification
- Open the CSV in Excel or pandas and inspect the score progression
- Plot centroid drift to confirm the sampler is exploring configuration space
