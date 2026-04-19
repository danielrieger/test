# Implementation Plan - Project Reorganization (Standard Src-Layout)

This plan moves the project from a fragmented structure to a **Standard Src-Layout**. This is the professional long-term fix for your import issues, ensuring that `import smlm_score` works consistently across WSL, Windows, and any future environments.

## User Review Required

> [!WARNING]
> - **Import Refactoring**: This is a significant change. We will need to update every file in `examples/` and `tests/` to use the new `smlm_score.<package>` prefix (e.g., `from smlm_score.imp_modeling import ...`).
> - **Git Conflicts**: If you have pending local changes, please commit them before we start.

## Step-by-Step Execution Guide (WSL Prompts)

This guide assumes you are in your project root: `cd ~/Thesis/smlm_score`

### Step 1: Create the new structure
```bash
# Create the top-level package and its init file
mkdir -p src/smlm_score
touch src/smlm_score/__init__.py

# Move existing sub-packages into the new root
mv src/imp_modeling src/smlm_score/
mv src/simulation src/smlm_score/
mv src/utility src/smlm_score/
mv src/validation src/smlm_score/
mv src/benchmarking src/smlm_score/
```

### Step 2: Update project configuration
You will need to update `pyproject.toml`. I can do this automatically, but the goal is to make `setuptools` aware of the `src` layout:
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["smlm_score*"]
```

### Step 3: Refactor Imports (The "Bulk Update")
We need to replace old import paths with the new absolute ones. 

**Old patterns to replace**:
- `from imp_modeling ...` -> `from smlm_score.imp_modeling ...`
- `from utility ...` -> `from smlm_score.utility ...`
- `from simulation ...` -> `from smlm_score.simulation ...`

**Bash helper to update current directory files**:
```bash
# Example for one prefix (run carefully or let me do it via tools)
find . -name "*.py" -not -path "*/.*" -exec sed -i 's/from imp_modeling/from smlm_score.imp_modeling/g' {} +
```

### Step 4: Re-install and Verify
```bash
# Re-install the project so the new paths are registered in the environment
pip install -e .

# Run verification
python verify_wsl_env.py
```

## Verification Plan

### Automated Tests
- Re-run `pip install -e .` in the WSL environment.
- Run the diagnostic script `verify_wsl_env.py` to confirm `smlm_score.__file__` is now valid.
- Execute one example (e.g., `NPC_example_BD.py`) to confirm imports are resolved.

## Open Questions

> [!QUESTION]
> 1. Would you like me to execute these steps for you now? It's much safer as I can double-check the import replacements as I go.
