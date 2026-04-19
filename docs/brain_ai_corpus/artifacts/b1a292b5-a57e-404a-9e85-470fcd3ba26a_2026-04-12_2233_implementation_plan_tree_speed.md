# Accelerate Tree Score: Numba Vectorization

## Problem

The Tree scorer (`computescoretree` / `computescoretree_with_grad`) is the slowest scoring function in the pipeline. The root cause is a **pure-Python double loop** iterating over ~1,000 data points × 8 model points per evaluation:

```python
# tree_score.py, line 180
for data_idx in range(len(dataxyz)):        # ~1,000 iterations in Python
    ...
    for local_idx, model_idx in enumerate(candidate_models):  # ~8 per data point
        diff = modelxyzs_query[int(model_idx)] - x_d
        exponent = -0.5 * np.dot(diff.T, np.dot(inv_sigma, diff))  # NumPy call overhead
```

In contrast, the Distance scorer uses **Numba `@jit(nopython=True)`** to compile an identical mathematical loop to machine code, making it ~100× faster on the same data.

### Benchmark Context

From [benchmark_scoring.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/benchmark_scoring.py), at N=1,000 data points (typical NPC cluster size):

| Scorer | Per-Step Time | Notes |
|--------|-------------|-------|
| Distance (Numba) | ~0.5 ms | `@jit(nopython=True, fastmath=True)` |
| Tree (Python) | ~50 ms | Pure Python + NumPy overhead |
| GMM (Numba) | ~0.01 ms | K≪N, only ~8 GMM components |

The Tree scorer is **~100× slower** than Distance despite computing the **exact same math** (when `n_model <= 64`, it doesn't even prune — it evaluates all model points per data point).

## Root Cause Analysis

Three compounding issues:

1. **No Numba compilation**: The inner loops are plain Python, so every iteration pays interpreter overhead + NumPy dispatch overhead for tiny arrays.
2. **Redundant covariance work**: `_prepare_distance_terms` already precomputes `inv_sigmas` and `log_prefactors`, which is good — but the loop body still creates temporary arrays (`np.zeros`, list indexing) in Python.
3. **Unnecessary indirection**: The `model_candidates_per_data` list-of-lists structure prevents Numba compilation (Numba cannot handle ragged Python lists).

## Proposed Changes

The strategy is to **rewrite the hot loop as a Numba-JIT kernel**, exactly mirroring the architecture that already makes Distance scoring fast.

> [!IMPORTANT]
> Since `n_model <= 64` (our NPC has 8 AV points), Tree scoring currently evaluates **all** model points for every data point — mathematically identical to Distance scoring. The only difference is Tree uses pure Python while Distance uses Numba. This plan eliminates that gap.

---

### Component 1: Core Numba Kernels

#### [MODIFY] [tree_score.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/smlm_score/imp_modeling/scoring/tree_score.py)

**Add two new Numba-JIT functions** that replace the inner Python loops:

1. **`_compute_tree_score_numba(dataxyz, inv_sigmas, log_prefactors, modelxyzs, candidate_starts, candidate_indices)`**
   - `@jit(nopython=True, fastmath=True)` compiled
   - Outer loop over data points, inner loop over candidate model indices
   - Uses the **flat CSR-style** representation of candidates (two arrays instead of list-of-lists) so Numba can handle it
   - Computes the same log-sum-exp as the current Python loop

2. **`_compute_tree_score_and_grad_numba(...)`**
   - Same as above but also computes and accumulates gradients
   - Returns `(score, grad)` tuple

**Flatten the candidate structure** for Numba compatibility:
```python
# Before (Python list-of-lists, Numba-incompatible):
model_candidates_per_data = [[0,1,2], [1,3], [0,1,2,3], ...]

# After (CSR-style flat arrays, Numba-compatible):
candidate_indices = np.array([0,1,2, 1,3, 0,1,2,3, ...])
candidate_starts  = np.array([0, 3, 5, 9, ...])  # start offset per data point
```

**Modify `computescoretree` and `computescoretree_with_grad`** to:
1. Keep the existing `_prepare_distance_terms` and `_build_model_candidates_per_data` calls unchanged
2. Convert the list-of-lists to CSR flat arrays
3. Call the new Numba kernel instead of the Python loop

---

### Component 2: Gradient Kernel

The `computescoretree_with_grad` function has the same problem — a pure Python loop with per-element gradient accumulation. The new `_compute_tree_score_and_grad_numba` kernel handles this in compiled code.

---

### Component 3: Benchmark Update

#### [MODIFY] [benchmark_scoring.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/benchmark_scoring.py)

- Add a Numba warmup call for the new Tree kernel (matching the existing warmup pattern)
- No other benchmark changes needed — the existing `computescoretree()` call will automatically use the new fast path

---

## What Does NOT Change

- **Mathematical output**: The score and gradient values remain bit-identical (same log-sum-exp formula)
- **KD-Tree pruning logic**: `_build_model_candidates_per_data` is untouched
- **API surface**: `computescoretree()` and `computescoretree_with_grad()` keep identical signatures
- **CUDA kernels**: Distance/GMM GPU paths are unaffected
- **Restraint wrappers**: `ScoringRestraintTree` calls the same functions

## Open Questions

> [!IMPORTANT]
> **Covariance structure**: The current Tree scorer calls `_extract_covariance_matrix` per data point, supporting mixed scalar/vector/matrix variance inputs. The Distance scorer always expects pre-built `(N, 3, 3)` covariance arrays. Should the Tree scorer **also** require pre-built covariance arrays (simplifying the Numba kernel), or should we keep the flexible variance input and convert at the boundary?
>
> My recommendation: Convert at the boundary in `_prepare_distance_terms` (which already does this) and pass the pre-built arrays to Numba. This is what the current code does, so no API change is needed.

## Verification Plan

### Automated Tests

1. **Correctness**: Run a script that compares `computescoretree()` output before and after the change on the same synthetic data. Assert scores match to within floating-point tolerance (`rtol=1e-10`).
2. **Gradient correctness**: Compare `computescoretree_with_grad()` gradients before/after.
3. **Benchmark**: Run `benchmark_scoring.py` and verify Tree eval time drops from ~50ms to ~0.5ms at N=1,000.

### Manual Verification

1. Run the full `NPC_example_BD.py` pipeline and confirm identical scoring output.
2. Compare benchmark plots before/after.

## Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|-------------------|
| Tree eval @ N=1,000 | ~50 ms | ~0.5–1.0 ms |
| Tree eval @ N=10,000 | ~500 ms | ~5–10 ms |
| Speedup factor | — | **50–100×** |
| First-call overhead | 0 ms | ~200 ms (Numba JIT compilation, one-time) |
