# Architectural Analysis: Integrative Modeling Platform (IMP)

This report provides a formal analysis of the **Integrative Modeling Platform (IMP)** repository, specifically focusing on the architectural components relevant to the **SMLM-IMP** thesis.

---

## 1. Structural Representation: The Model & Particle Engine

The IMP core (found in the `kernel` module) follows a strictly decoupled design between data storage and functional representation.

### 1.1 Model and Particle (Data Store)
The `IMP::Model` acts as a centralized attribute database. A `IMP::Particle` does not store data itself; instead, it is a lightweight handle (containing a `ParticleIndex`) pointing to a specific row in the `Model`'s attribute tables (Float, Int, String, etc.).

> [!TIP]
> This design enables O(1) access to attributes and ensures that memory for structural data is managed contiguously, which is critical for the performance of large-scale models like the NPC.

### 1.2 The Decorator Pattern
Since particles are generic rows in a table, IMP uses **Decorators** to provide type-safe APIs for specific physical properties.
- **`XYZ`**: Decorates a particle with Cartesian coordinates and a radius.
- **`Mass`**: Decorates a particle with mass, facilitating rigid body calculations.
- **`Hierarchy`**: (Located in `modules/atom`) This is the most critical decorator for the thesis. It allows particles to have parents and children, enabling **multiscale modeling** where the same biological entity can be represented as a single bead or an atomistic ensemble simultaneously.

---

## 2. Theoretical Framework: Scoring & Restraints

Scoring in IMP is handled through the **Restraint** system, which translates spatial configurations into numerical energy values (or log-likelihoods).

### 2.1 The Scoring Pipeline
1. **`Restraint`**: A class that evaluates a specific score based on a set of particles (e.g., a ConnectivityRestraint or an SMLM Bayesian score).
2. **`ScoringFunction`**: An aggregator that sums multiple restraints.
3. **`Model::evaluate()`**: The core entry point that iterates through all active restraints to compute the global model objective.

### 2.2 Bayesian Integration
In the SMLM-IMP pipeline, the **`ScoringRestraintWrapper`** acts as a bridge, wrapping the custom GMM/Tree/Distance kernels into standard IMP `Restraint` objects. This allows the high-level samplers (REMC) to "see" the experimental data as standard energy potentials.

---

## 3. Sampling Engine: Replica Exchange Monte Carlo (REMC)

The Bayesian sampling workflow is orchestrated by the `pmi` module through the `ReplicaExchange` macro.

### 3.1 Metropolis-Hastings Implementation
The lower-level sampling logic in `IMP::core::MonteCarlo` implements the standard Metropolis criterion for move acceptance:
- **Downhill Move**: $\Delta E < 0 \implies$ Always accepted.
- **Uphill Move**: $\Delta E > 0 \implies$ Accepted with probability $P$:
  $$P(\text{accept}) = \min\left(1, e^{-\frac{\Delta E}{k_B T}}\right)$$

### 3.2 Temperature Swaps
The `ReplicaExchange` macro manages a population of replicas at different temperatures ($T$). By periodically attempting to swap temperatures between adjacent replicas, the system avoids getting trapped in local minima—a process essential for capturing the structural heterogeneity of the NPC.

---

## 4. Simulation Engine: Brownian Dynamics (BD)

For geometric relaxation and structural fitting, the pipeline uses **Brownian Dynamics** (overdamped Langevin dynamics).

### 4.1 Implementation Logic
The implementation in `IMP::atom::BrownianDynamics.cpp` updates coordinates using a simple integration scheme that balances deterministic forces against thermal noise:

$$r(t + \Delta t) = r(t) + \frac{F(t) \Delta t}{\gamma} + \delta R(t)$$

Where:
- $F(t)$ is the deterministic force from the scoring function.
- $\gamma$ is the friction coefficient.
- $\delta R(t)$ is a random displacement with zero mean and variance satisfying the fluctuation-dissipation theorem: $\langle \delta R^2 \rangle = 2 D \Delta t$.

> [!NOTE]
> In our pipeline, BD is primarily used as a "geometric smoother" to resolve steric clashes (excluded volume) before entering the more computationally intensive REMC sampling phase.

---

## 5. Persistence Layer: Rich Molecular Format (RMF)

The `rmf` module handles the serialization of IMP models into a compressed, hierarchical binary format.

- **Hierarchy-Centric**: RMF is designed to store the PDB/mmCIF hierarchy. 
- **Standalone Objects**: Particles that are not formally children of the root hierarchy (like our Accessible Volume particles) must be manually tracked or attached to the hierarchy to be recorded in the trajectory. Our recently implemented "AV trajectory" writer bypasses this by parsing the `stat` files and reconstructing the movement manually into a dedicated RMF3 file.
