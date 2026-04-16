# SMLM-IMP Scoring Performance Summary
| Metric | Distance | Tree (Opt) | GMM |
| :--- | :---: | :---: | :---: |
| Eval Latency (N=1k) | 141.563 ms | 42.654 ms | 0.284 ms |
| Eval Latency (N=10k) | 1533.350 ms | 458.933 ms | 0.285 ms |
| Initialization (N=1k) | 0.00 ms | 0.94 ms | 56.63 ms |
| Total MCMC (10k steps) | 1415.63 s | 426.54 s | 2.90 s |
| Speedup (vs Distance @ 10k) | 1.0x | 3.3x | 5382.1x |