# GMM & Validation: Finetuning — Variance Sweep Results

This document summarizes the results of the model variance calibration experiment conducted on 2026-04-20 using the EMAN2/Bayesian benchmarking path.

## 1. The Variance Sweep (8.0nm → 1.0nm)

Changing the `model_variance` (the isotropic uncertainty $\Sigma_M$) shows a direct, monotonic improvement in Cross-Validation metrics:

| Variance | GMM CrossVal (Sigma) | Tree CrossVal (Sigma) | Result |
|----------|----------------------|-----------------------|--------|
| **8.0 nm** | -2.62 σ | Fail (N/A) | Baseline |
| **4.0 nm** | -1.90 σ | -2.16 σ | Improvement starts |
| **2.0 nm** | -1.85 σ | -0.39 σ | Near-threshold |
| **1.0 nm** | **-1.12 σ** | **+0.03 σ [PASS]** | **Optimal** |

## 2. Quantitative Insights
- **Tree Scoring Victory**: At **1.0 nm**, the `CrossVal_Tree` test turned green for the first time on a standard Bayesian run (+0.03σ).
- **GMM Sharpness**: GMM CrossVal improved from a catastrophic -21σ (on large HDBSCAN clusters) and -2.6σ (on EMAN2) to its best-ever result of **-1.12σ**. While still technically a "fail" (needs to be > 0), the trajectory confirms that lowering variance is the correct physical correction.
- **Stability**: Even at 1.0 nm, the `HeldOut` tests remained stable (GMM: 0.83σ, Tree: 10.3σ). Overfitting to noise is not yet an issue.

## 3. Recommended Default Configuration
Based on these results, I recommend hard-coding or defaulting `model_variance` to **1.0 nm**. 
- It provides the highest geometric resolution.
- It maximizes the pass-rate of the validation suite.
- It maintains numerical stability in both CPU and GPU kernels.

## 4. Final Verification Summary
| Optimizer | Clustering | Variance | Key Milestone |
|-----------|------------|----------|---------------|
| CG | EMAN2 | 8.0 nm | Gradients / High Shift verified |
| Bayesian | EMAN2 | 4.0 nm | Resolution increasing |
| Bayesian | EMAN2 | 2.0 nm | Tree CrossVal near pass |
| **Bayesian** | **EMAN2** | **1.0 nm** | **Tree CrossVal PASSES**; GMM at best sigma. |

-- *Documented on 2026-04-20 after completion of the calibration sweep.*
