# EMAN2 Particle Picking & High-Res Modeling Workflow

This document details the high-resolution particle picking workflow that integrates EMAN2's neural network autoboxing with the SMLM-IMP scoring pipeline.

## 1. High-Resolution MRC Generation

The standard SMLM dataset is converted into a high-resolution 2D density map (MRC format) to resolve the 8-fold symmetry of the NPC.

**Script:** `examples/smlm_to_mrc.py`
- **Resolution:** 5.0 nm/pixel (Upgraded from 10nm)
- **Weighting:** Intensity-weighted using localization amplitudes (`Amplitude_0_0`)
- **Output:** `examples/micrograph.mrc` and `examples/pixel_map.json`

## 2. EMAN2 Neural Network Picking

The rendered micrograph is processed in EMAN2 to isolate high-quality NPC candidates.

### Stage A: Training (GUI)
1. Open the EMAN2 boxer: `e2boxer.py micrograph.mrc --apix=50.0 --boxsize=64`
2. Train a Convolutional Neural Network (CNN) by selecting ~50 "Good" (complete rings) and ~100 "Bad" (noise, overlaps) examples.
3. Save the trained model to the project database.

### Stage B: Headless Picking (CLI)
Once trained, particles can be picked automatically across the full map:
```bash
e2boxer.py micrograph.mrc --autopick=nn:threshold=0.5 --write_ptcls
```
This generates `examples/info/micrograph_info.json` containing the center coordinates for all picks.

## 3. Targeted IMP Modeling

The pipeline consumes the EMAN2 coordinates to perform precise structural fitting.

**Configuration:** `pipeline_config.json`
```json
"clustering": {
    "method": "eman2",
    "eman2_boxes": "info/micrograph_info.json",
    "pixel_map": "pixel_map.json"
}
```

**Optimization modes used:**
- **Frequentist (MLE):** Rapid Conjugate Gradients alignment for screening 300+ particles.
- **Bayesian (REMC):** Full posterior sampling (200 frames) to generate structural trajectories and uncertainty maps.

## 4. Final Results: The "Money Shot"

The workflow produces publication-quality figures showing the alignment of the 8-fold symmetric protein model with the raw localization density.

![Final Thesis Result: Box 240](https://github.com/danielrieger/test/blob/main/examples/figures/methodology/thesis_final_result_v4.png?raw=true)

### Key Performance Metrics:
- **Spatial Resolution:** 5nm
- **Validation Accuracy:** 3/4 tests passed (Separation, Cross-Val Tree, Held-Out)
- **Evaluation Speed:** Constant-time $O(GK)$ using the GMM engine.
