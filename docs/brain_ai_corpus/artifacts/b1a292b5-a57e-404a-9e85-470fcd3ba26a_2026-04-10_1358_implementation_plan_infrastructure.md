# Infrastructure Upgrade — WSL2 Migration & EMAN2

## Problem Statement

The current development environment (`C:\envs\py311`) was transplanted from another machine via `conda-pack`. This causes:

1. **No package manager** — `conda` itself is absent; you cannot install or update packages.
2. **Hardcoded DLL paths** — 30+ IMP native C++ libraries (`imp_kernel.dll`, `imp_atom.dll`, etc.) plus ~200 system DLLs in `Library\bin` have baked-in Windows paths from the source machine.
3. **No MPI** — `mpi4py` is missing, so `ReplicaExchange` runs in serial mode (single-replica fallback).
4. **No working CUDA** — `numba.cuda.is_available()` returns `False` despite CUDA 12 pip packages being present. The T500 GPU is unused.
5. **EMAN2 requirement** — Native Windows binaries were discontinued (July 2025). The only supported installation path on Windows is via **WSL2** using `conda create -n eman2 eman-dev -c cryoem -c conda-forge`. ([Source: BCM CryoEM](https://cryoem.bcm.edu/cryoem/downloads/view_eman2_versions))

## Terminal Guide

| Step | Terminal | Why |
|:---|:---|:---|
| Step 1 (WSL install) | **PowerShell (Run as Administrator)** | System-level Windows feature |
| Steps 2–8 | **Ubuntu terminal** (via Start menu or `wsl` command) | All Linux commands |

## Current Environment Audit

### Critical Packages (Must Reproduce Exactly)

| Package | Version | Source | Notes |
|:---|:---|:---|:---|
| **Python** | 3.11 | conda | Base interpreter |
| **IMP** | 2.22.0 | conda-forge | 30+ native `.dll`/`.so` files |
| **IMP.bff** | (bundled) | conda-forge | Accessible Volume computation |
| **IMP.pmi** | (bundled) | conda-forge | PMI macro layer |
| **ihm** | 2.3 | conda-forge | mmCIF/IHM data model |
| **RMF** | (bundled) | conda-forge | `RMF.dll` present |
| **numpy** | 2.1.3 | pip | |
| **scipy** | 1.17.1 | pip | |
| **scikit-learn** | 1.8.0 | pip | HDBSCAN via `sklearn.cluster` |
| **matplotlib** | 3.10.1 | pip | |
| **pandas** | 2.2.3 | pip | |
| **numba** | 0.61.0 | pip | JIT compilation |
| **plotly** | 6.0.1 | pip | Interactive visualization |
| **opencv-python** | 4.11.0 | pip | |

### Development & Tooling Packages

| Package | Version | Notes |
|:---|:---|:---|
| **jupyter/jupyterlab** | 4.3.6 | Notebook environment |
| **pytest** | 9.0.2 | Test runner |
| **tqdm** | 4.67.1 | Progress bars |
| **mcp** | 1.26.0 | Model Context Protocol (Antigravity) |
| **pydantic** | 2.12.5 | Data validation |
| **uvicorn/starlette** | 0.42.0 / 1.0.0 | MCP server runtime |

### Missing (Should Be Added in Rebuild)

| Package | Why |
|:---|:---|
| **mpi4py** | Enables true parallel Replica Exchange (multi-replica MCMC) |
| **hdbscan** | Standalone library (currently using sklearn's bundled version, which is fine) |

### Windows-Only (Will Not Migrate)

| Package | Replacement |
|:---|:---|
| `pywin32` | Not needed on Linux |
| `pywinpty` | Linux PTY is native |
| `win_inet_pton` | Built into Linux |
| `PySide6` + `shiboken6` | Will install Linux builds; GUI via WSLg |

## User-Specific Configuration

| Parameter | Value |
|:---|:---|
| **GPU** | NVIDIA Quadro T500 (Turing, CUDA-capable in WSL2) |
| **IDE** | PyCharm Professional (full WSL remote interpreter support) |
| **Disk** | 250 GB free on C: (ample for WSL virtual disk) |
| **Tooling** | Antigravity + Codex (remain on Windows, access WSL filesystem) |

> [!IMPORTANT]
> **Filesystem Performance**: WSL2 accessing `/mnt/c/` (your Windows drive) is **~10x slower** for Git, indexing, and small-file I/O due to the 9p translation protocol. The project should live on the native Linux EXT4 filesystem (`~/Thesis/`), accessible from Windows via `\\wsl$\Ubuntu\home\<user>\Thesis`.

> [!WARNING]
> **OneDrive Sync**: Your current project is inside `OneDrive\Desktop`. The WSL copy will **not** be synced to OneDrive. You should use **Git** (pushing to `danielrieger/test`) as your backup and sync mechanism instead.

## Migration Path

### Step 1: Enable WSL2 (requires admin + reboot)

```powershell
# Run in an ADMINISTRATIVE PowerShell
wsl --install
# This installs WSL2 + Ubuntu by default
# Reboot when prompted
```

After reboot, Ubuntu will auto-launch and ask you to create a Linux username/password.

### Step 2: Verify GPU Passthrough

```bash
# Inside WSL terminal
nvidia-smi
# Should show Quadro T500 with driver version
```

> [!NOTE]
> WSL2 uses the **Windows NVIDIA driver** — you do NOT install a separate Linux driver. Just ensure your Windows driver is recent (Game Ready or Studio, 535+ recommended).

### Step 3: Install Miniforge

```bash
# Inside WSL
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Step 4: Clone Project into Linux Filesystem

```bash
mkdir -p ~/Thesis
cd ~/Thesis
git clone https://github.com/danielrieger/test.git smlm_score
git clone --depth 1 https://github.com/danielrieger/imp.git imp_source
```

### Step 5: Create SMLM-IMP Environment

```bash
# Create the base environment with IMP, MPI, and full CUDA toolkit support
# Added cuda-toolkit to ensure the compiler and libraries are available for Numba
conda create -n smlm python=3.11 imp=2.22 ihm mpi4py cuda-toolkit cuda-version=12 -c conda-forge -y
conda activate smlm

# Install pip packages with flexible versions for Linux compatibility
# We use pypi for these to match the exact versions from your Windows setup
pip install "numpy>=2.1.3" "scipy>=1.17.1" "scikit-learn>=1.8.0" \
  "matplotlib>=3.10.1" "pandas>=2.2.3" "numba>=0.61.0" "plotly>=6.0.1" \
  "opencv-python-headless>=4.11.0" tqdm pytest jupyterlab pydantic \
  mcp uvicorn starlette pyyaml

# Install your project in editable mode
cd ~/Thesis/smlm_score
pip install -e .
```

### Step 6: Create EMAN2 Environment (Separate)

```bash
conda create -n eman2 eman-dev -c cryoem -c conda-forge -y
```

### Step 7: Configure PyCharm Professional

1. Open PyCharm on Windows.
2. **File → Open** → navigate to `\\wsl$\Ubuntu\home\<user>\Thesis\smlm_score`.
3. **Settings → Python Interpreter → Add → WSL** → select the Ubuntu distribution → point to `~/miniforge3/envs/smlm/bin/python`.
4. PyCharm will index the project using the Linux interpreter — all imports, debugging, and test execution will run natively in WSL.

### Step 8: Configure Antigravity & Codex

Both tools should be pointed to the WSL filesystem path:
- **Workspace**: `\\wsl$\Ubuntu\home\<user>\Thesis\smlm_score`
- **Python**: The WSL interpreter (accessed via `wsl` commands or direct path)

## Rollback Strategy

> [!TIP]
> The current Windows environment at `C:\envs\py311` is **not modified or deleted** by this plan. If anything goes wrong, you can always fall back to the existing setup. We only delete the old environment after the new one is validated.

## Verification Plan

| Check | Command | Expected Result |
|:---|:---|:---|
| GPU visible | `nvidia-smi` (in WSL) | Shows Quadro T500 |
| IMP loads | `python -c "import IMP; print(IMP.__version__)"` | `2.22.0` |
| IMP.bff loads | `python -c "import IMP.bff"` | No error |
| MPI works | `python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"` | `1` (single process) |
| CUDA works | `python -c "import numba.cuda; print(numba.cuda.is_available())"` | `True` |
| Scoring parity | `cd ~/Thesis/smlm_score && python tests/validate_tree_optimization.py` | 14-decimal match |
| Pipeline runs | `cd examples && python NPC_example_BD.py` | 6/6 validations pass |
| EMAN2 GUI | `conda activate eman2 && e2projectmanager.py` | GUI window appears via WSLg |
