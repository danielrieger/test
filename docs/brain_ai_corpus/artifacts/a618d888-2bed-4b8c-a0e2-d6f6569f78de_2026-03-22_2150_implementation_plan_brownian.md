# Brownian Dynamics Integration Plan

The pipeline currently supports:
1. **Bayesian Mode (MCMC)**: Uses Monte Carlo sampling to explore the posterior distribution. Needs no gradients (works with GMM).
2. **Frequentist Mode (CG)**: Uses `ConjugateGradients` to find the exact local minimum. Relies on analytical gradients (Tree/Distance).

We will add a third mode:
3. **Brownian Mode (BD)**: Uses `IMP.atom.BrownianDynamics`. It combines the gradient-following of the Frequentist mode with temperature-based random perturbations, simulating the physical "pull" of the data on the molecular structure over time. Like Frequentist mode, it requires the analytical gradients we just implemented for Tree and Distance scores.

## Architecture

```mermaid
graph TD
    A[PCA-Aligned Model] --> B{OPTIMIZATION_MODE}
    B -->|"bayesian"| C[MCMC Sampler (No Gradients)]
    B -->|"frequentist"| D[ConjugateGradients (Needs Gradients)]
    B -->|"brownian"| E[BrownianDynamics (Needs Gradients)]
    
    E --> F[ScoringRestraint_unprotected_evaluate]
    F -->|Return Forces| E
    E -->|Add Thermal Noise| G[Update Particle Positions dt]
```

## Proposed Changes

### [MODIFY] [pipeline_config.json](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/pipeline_config.json)
Add a configuration sub-block for Brownian dynamics under the `optimization` section:
```json
"brownian": {
    "scoring_type": "Tree",
    "temperature_k": 300.0,
    "max_time_step_fs": 50000.0,
    "number_of_bd_steps": 500,
    "rmf_save_interval": 10
}
```

### [MODIFY] [NPC_example_BD.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/examples/NPC_example_BD.py)
Add the `elif OPTIMIZATION_MODE == "brownian"` branch. Call the existing `run_brownian_dynamics_simulation` function from `simulation_setup.py`. We essentially just hook up the configuration settings to the function arguments.

### [MODIFY] [simulation_setup.py](file:///c:/Users/User/OneDrive/Desktop/Thesis/smlm_score/src/imp_modeling/brownian_dynamics/simulation_setup.py)
The existing `run_brownian_dynamics_simulation` is well-written but requires minor cleanup to match the signature style of `run_bayesian_sampling` and `run_frequentist_optimization`.
- Pass `output_dir` as an argument.
- Use `scoring_restraint_wrapper` correctly.
- Ensure mass is configured for particles (required for Brownian dynamics physics calculations).

## Verification
- Run `NPC_example_BD.py` with `OPTIMIZATION_MODE = "brownian"`.
- Verify that a `bd_trajectory.rmf` is produced and that the score improves over the simulation steps.
