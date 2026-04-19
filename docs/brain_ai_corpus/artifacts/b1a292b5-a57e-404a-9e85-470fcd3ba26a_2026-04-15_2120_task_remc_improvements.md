# Task: REMC Performance & Configuration Improvements

Improve Bayesian sampling efficiency by implementing score scaling and exposing rigid body move parameters.

- `[x]` **Phase 1: Configuration Updates**
    - `[x]` Add `score_weight`, `max_rb_trans`, `max_rb_rot` to `pipeline_config.json`.
    - `[x]` Increase sampling depth (`number_of_frames`, `monte_carlo_steps`) in `pipeline_config.json`.
- `[x]` **Phase 2: Logic Implementation**
    - `[x]` Update `mcmc_sampler.py` to handle `score_weight` and mover parameters.
    - `[x]` Update `NPC_example_BD.py` to calculate `"auto"` weight (1/N).
- `[x]` **Phase 3: Progress & Logging**
    - `[x]` Add frame-level progress reporting to `mcmc_sampler.py` to prevent "frozen" perception.
- `[x]` **Phase 4: Verification**
    - `[x]` Run a benchmark to verify improved REMC acceptance rates.
