# SMLM-IMP Scoring Performance Summary
| Metric | Distance | Tree (Opt) | GMM |
| :--- | :---: | :---: | :---: |
| Eval Latency (N=1k) | 139.879 ms | 41.462 ms | 0.009 ms |
| Eval Latency (N=10k) | 1528.025 ms | 480.506 ms | 0.019 ms |
| Initialization (N=1k) | 0.00 ms | 0.42 ms | 54.27 ms |
| Total MCMC (10k steps) | 1398.79 s | 414.62 s | 0.15 s |
| Speedup (vs Distance @ 10k) | 1.0x | 3.2x | 78386.3x |