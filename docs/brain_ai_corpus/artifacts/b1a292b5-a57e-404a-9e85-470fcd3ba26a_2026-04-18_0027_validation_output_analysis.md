# Analysis of SMLM-IMP Validation Outputs

You asked why certain tests were missing, why the map was filtered, and where the 200 REMC sampling steps went. **The mathematical and structural flow of the code did not change during the refactoring.** The differences between the old terminal output you remember and the recent run are entirely determined by the `pipeline_config.json` configuration driving the pipeline.

### 1. "Why filtered? We introduced full map before?"

The pipeline is correctly responding to the current state of `examples/pipeline_config.json`.
In your config file, the filtering mode is explicitly set to `"random"`:
```json
"filtering": {
    "type": "random",
    "random": {
        "size_percentage": 15
    }
}
```
When this is set to `"random"`, the pipeline isolates a 15% random sub-window of the SMLM data to process. This means **the specific clusters found (like your old Cluster 87) will change on every single run**. To restore the "full map" processing, you must simply update your configuration file: `"type": "none"`.

### 2. "Where is REMC and the 200 sampling steps?"

The Bayesian (REMC) optimization is intentionally missing from the terminal output because it **never triggered**.

In your `pipeline_config.json`:
```json
"execution": {
    "target_cluster_id": null
}
```
Currently, the pipeline has no specifically assigned valid cluster. When the script encounters `null`, it gracefully defaults to only scoring the 3 Noise clusters to verify background heuristics. The optimization trigger naturally bounds itself to skip noise clusters (as deploying REMC to fit a structure against random noise would be computationally wasteful and mathematically incorrect). 

If you provide a valid cluster ID (e.g., `"target_cluster_id": 87`) or allow the script to automatically target the first valid cluster, the REMC sequence will trigger and standard Bayesian progress logs will print exactly as they did before.

### 3. "Where are the other tests?"

The `HeldOut_Tree` and `HeldOut_GMM` validations are designed as forms of structural Cross-Validation. They rely on having an *optimized valid model* to test against the remaining out-of-sample data points.

Since no valid cluster was evaluated (due to `target_cluster_id` being `null`), no model was optimized. The pipeline is designed to organically shrink the validation suite depending on the available data. Without a valid target to run CV on, it intelligently skips the HeldOut tests and only generates the `Separation` tests based on the noise pools it was able to evaluate.

---

### Conclusion

The refactoring did exactly what it was supposed to do: it solidified the code architecture while rigidly honoring the `pipeline_config.json` file. If you adjust your config back to your previous settings (`"type": "none"` and choosing a valid cluster), the exact 200-step REMC logs and all 4 validation tests will return.
