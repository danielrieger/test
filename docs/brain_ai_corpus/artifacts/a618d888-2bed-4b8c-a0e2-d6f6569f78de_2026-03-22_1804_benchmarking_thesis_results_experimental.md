# Scoring Benchmarks on Real Experimental SMLM Data

This benchmark repeats the pure computational tests, but this time extracts precise subsets from your actual `ShareLoc_Data/data.csv` bounding box (10k-12k x, 0-5k y). This proves to your thesis readers that the synthetic noise scaling perfectly aligns with the real-world cluster structures handled by the pipeline.

## Figure C: Experimental Evaluation Scaling (Log-Log)
Using the real dataset (ranging up to the maximum 29,472 points found in the bounding box), we again clearly see the distinct mathematical complexity classes.

![Experimental Scaling](file:///C:/Users/User/.gemini/antigravity/brain/a618d888-2bed-4b8c-a0e2-d6f6569f78de/bench_figC_exp_scaling.png)

> [!NOTE]
> Even on real data, GMM retains its completely flat $\mathcal{O}(GK)$ scaling profile. Evaluating the entire 29,000+ points simultaneously takes $\approx 0.1$ ms per evaluation.

## Figure D: The MCMC Sampling Trade-off (1 Step vs 10,000 Steps)
To address the exact question: **"Does it show that the whole potential of GMM is only seeable at the end, meaning MCMC?"**, this figure compares the total time cost of running exactly *1 evaluation* against running *10,000 MCMC evaluations* for a 1,000 point extracted cluster.

![Experimental MCMC Tradeoff](file:///C:/Users/User/.gemini/antigravity/brain/a618d888-2bed-4b8c-a0e2-d6f6569f78de/bench_figD_exp_tradeoff.png)

> [!IMPORTANT]
> **Subplot A** visualizes the answer: If we were only scoring the structure once, the massive initialisation cost of fitting the Gaussian Mixture makes GMM the absolute **worst** choice (taking ~100ms compared to Distance's ~40ms and Tree's practically instant start).
> 
> However, **Subplot B** proves why GMM dominates your pipeline. Because MCMC requires thousands of rapid evaluations, the upfront 100ms setup cost becomes completely invisible. By the time 10,000 evaluations finish, GMM processes data drastically faster than its $\mathcal{O}(NM)$ Distance counterpart.
