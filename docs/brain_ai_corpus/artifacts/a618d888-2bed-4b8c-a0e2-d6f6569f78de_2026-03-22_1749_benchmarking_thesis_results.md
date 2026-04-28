# Scoring Functions Computational Benchmarks

Here are the benchmarking results comparing **Distance** $\mathcal{O}(NM)$, **Tree** $\mathcal{O}(N \log M)$, and **GMM** $\mathcal{O}(NK)$ scoring methods. The benchmarks bypass IMP overhead entirely and directly measure the mathematical execution times as tested in `tests/test_scoring.py` during your MCMC sampling.

## Figure A: Per-Step Scaling (Time vs Data Size)
This log-log plot demonstrates the empirical scaling behaviors of the three methods as the raw SMLM dataset grows from extremely sparse (100 points) to extremely dense (10,000 points).

![Scoring Engine Computational Scaling](file:///C:/Users/User/.gemini/antigravity/brain/a618d888-2bed-4b8c-a0e2-d6f6569f78de/bench_figA_scaling.png)

> [!TIP]
> Notice how GMM's evaluation time is completely flat? Because the evaluation $\mathcal{O}(GK)$ only scales with the number of *Gaussians* ($K \le 8$), not the number of SMLM points ($N$). Once the model is fitted, scoring a 10,000 point NPC takes the exact same sub-millisecond time as a 100 point NPC.

---

## Figure B: Performance Trade-off (Init vs Eval Cost)
While GMM evaluation is incredibly fast, it requires fitting a Gaussian Mixture Model upfront using the Bayesian Information Criterion (BIC), which is computationally heavy. This chart looks specifically at a **1,000 point NPC** geometry, simulating the cost ratio of running exactly 10,000 MCMC optimization steps.

![Performance Trade-off](file:///C:/Users/User/.gemini/antigravity/brain/a618d888-2bed-4b8c-a0e2-d6f6569f78de/bench_figB_tradeoff.png)

> [!NOTE]
> Even when accounting for its expensive initialization time (grey bar), the **GMM method completely dominates** over the total lifespan of an optimization run because of its $\approx0.1$ ms per-eval runtime. For a full simulation of $1\times 10^5$ steps, GMM saves minutes (or hours on thousands of NPCs) compared to the linear Distance baseline.
