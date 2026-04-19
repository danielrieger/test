# Implementation Plan — Environment Stabilization (Optimized)

The audit found **60 missing packages** and **70+ version mismatches** between your Windows baseline and WSL. However, the vast majority are harmless. This plan triages them into actionable categories and provides a single, precise execution sequence.

## Triage Summary

| Category | Count | Action |
|:---|:---:|:---|
| **Windows-only** (skip) | ~15 | `pywin32`, `pywinpty`, `colorama`, `win-inet-pton`, `conda-pack`, `PySide6`, `shiboken6`, `nvidia-cuda-*` pip wheels | 
| **Conda-managed** (don't touch) | 3 | `numpy` (via conda), `ihm` 2.9, `rmf` 1.7 — tied to IMP 2.22.0 |
| **Critical scientific** (pin) | 4 | `numba`, `llvmlite`, `pandas`, `scipy` |
| **Minor bumps** (harmless) | ~50 | `matplotlib` 3.10.8 vs 3.10.1, `tqdm` 4.67.3 vs 4.67.1, etc. |
| **Genuinely missing** | ~5 | `numba-cuda`, `pytz`, `opencv-python-headless` version alignment |

## User Review Required

> [!IMPORTANT]
> **numpy stays at 2.4.3** — IMP 2.22.0 was built against it via conda-forge. Downgrading numpy via conda is technically possible (dry-run confirmed), but it risks ABI mismatches with IMP's compiled C++ extensions. Since your code uses `>=` specifiers and numpy 2.x maintains backward compat, keeping 2.4.3 is the safer choice.

> [!WARNING]  
> **numba 0.61.0 requires numpy < 2.2**. This creates a conflict: IMP wants numpy ≥ 2.4, numba 0.61 wants numpy < 2.2. The solution is to **keep numba at 0.65.0** (which supports numpy 2.4) and verify your CUDA kernels still work. The kernel API has not changed between 0.61 and 0.65.

## Proposed Changes

### Phase 1: Pin pandas and install missing packages
These are pip-only changes that won't affect conda packages:

```bash
conda activate smlm
pip install "pandas==2.2.3" "pytz>=2024.1" "numba-cuda>=0.28.2"
```

**Rationale:**
- `pandas` 3.0 introduced breaking API changes (e.g., `DataFrame.append()` removed). Your code was written against 2.2.x.
- `pytz` is a runtime dependency of pandas 2.2.x.
- `numba-cuda` provides the `numba.cuda` subpackage that enables GPU kernels.

### Phase 2: Verify — no further changes needed

The following version bumps are **harmless** and should be kept as-is:
- `scipy` 1.17.1 → still compatible (API-stable within major version)
- `scikit-learn` 1.8.0 → same version ✅
- `matplotlib` 3.10.8 vs 3.10.1 → minor patch
- `plotly` 6.6.0 vs 6.0.1 → minor patch
- `opencv-python-headless` 4.13 vs 4.11 → minor patch
- All Jupyter/IPython packages → newer is fine
- `ihm` 2.9 vs 2.3 → conda-managed, tied to IMP build

### What we explicitly skip

| Package | Reason |
|:---|:---|
| `PySide6`, `shiboken6` | Qt GUI framework — not needed in headless WSL |
| `pywin32`, `pywinpty` | Windows-only system bindings |
| `colorama`, `win-inet-pton` | Windows console/networking shims |
| `conda-pack` | Was the old packaging tool — no longer relevant |
| `nvidia-cuda-nvcc-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-runtime-cu12` | Replaced by conda's `cuda-toolkit` |
| `cuda-bindings`, `cuda-core`, `cuda-pathfinder` | Windows pip CUDA bindings, replaced by conda toolkit |
| `exceptiongroup`, `tomli`, `zipp`, `importlib-*` | Backports for Python < 3.11, unnecessary |

## Execution — Single Command

```bash
conda activate smlm
pip install "pandas==2.2.3" "pytz>=2024.1" "numba-cuda>=0.28.2"
```

That's it. One command.

## Verification Plan

```bash
# 1. Confirm core stack
python -c "
import numpy, numba, scipy, pandas, sklearn
from numba import cuda
print(f'numpy:   {numpy.__version__}')
print(f'numba:   {numba.__version__}')
print(f'scipy:   {scipy.__version__}')
print(f'pandas:  {pandas.__version__}')
print(f'sklearn: {sklearn.__version__}')
print(f'CUDA:    {cuda.is_available()}')
import IMP; print(f'IMP:     {IMP.get_module_version()}')
"

# 2. Smoke test the pipeline
cd ~/Thesis/smlm_score/examples
python NPC_example_BD.py --steps 10
```
