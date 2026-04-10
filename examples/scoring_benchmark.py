import time
import numpy as np
import pandas as pd
import IMP
import IMP.core
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Import project modules (Analog zu NPC_example_BD.py)
from smlm_score.imp_modeling.restraint.scoring_restraint import ScoringRestraintWrapper
from smlm_score.utility.input import read_parameters_from_json, read_experimental_data
from smlm_score.utility.data_handling import flexible_filter_smlm_data, compute_av
from smlm_score.imp_modeling.scoring.gmm_score import test_gmm_components


class ScoringBenchmark:
    """
    Klasse zur Durchführung von Effizienz-Benchmarks für verschiedene SMLM-Scoring-Methoden.
    Vergleicht: Distance (Wu et al.), GMM (Bonomi et al.), und Tree Scoring.
    """

    def __init__(self,
                 smlm_data_path: str = "ShareLoc_Data/data.csv",
                 pdb_data_path: str = "PDB_Data/7N85-assembly1.cif",
                 av_parameters_path: str = "av_parameter.json"):

        self.paths = {
            "smlm": smlm_data_path,
            "pdb": pdb_data_path,
            "av": av_parameters_path
        }
        self.data_cache = {}
        self.model_cache = {}

        print("--- Initializing Benchmark Suite ---")
        self._load_data_and_model()

    def _load_data_and_model(self):
        """Lädt Daten und IMP Modell einmalig, um I/O aus den Benchmarks auszuschließen."""
        print(f"Loading Parameters from {self.paths['av']}...")
        self.parameters = read_parameters_from_json(self.paths['av'])

        print(f"Loading SMLM Data from {self.paths['smlm']}...")
        self.raw_smlm_data = read_experimental_data(self.paths['smlm'])
        if self.raw_smlm_data is None:
            raise ValueError("Failed to load SMLM data.")

        print(f"Building IMP Model/AVs from {self.paths['pdb']}...")
        # Wir speichern avs und m, damit wir sie für jeden Test frisch übergeben können,
        # falls nötig, oder wiederverwenden.
        self.avs, self.m, self.pdb_hierarchy = compute_av(self.paths['pdb'], self.parameters)
        print("Initialization complete.\n")

    def _prepare_scorer(self, scoring_type: str, data_subset_fraction: float = 1.0):
        """
        Bereitet den Scorer vor (Setup-Phase).
        Gibt (wrapper, setup_time) zurück.
        """
        # Daten filtern (ggf. Subset für Scaling-Tests)
        current_data = self.raw_smlm_data
        if data_subset_fraction < 1.0:
            current_data = current_data.sample(frac=data_subset_fraction, random_state=42)

        # Zeitmessung Start (Setup)
        start_time = time.perf_counter()

        should_return_tree = (scoring_type == "Tree")

        # 1. Filter / Pre-processing
        smlm_coords, smlm_vars, smlm_coords_tree, kdtree_obj, _ = flexible_filter_smlm_data(
            current_data,
            filter_type='cut',
            x_cut=(10000, 12000),  # Analog NPC_example
            y_cut=(0, 5000),
            fill_z_value=0.0,
            return_tree=should_return_tree
        )

        # 2. Spezifische Vorbereitung (GMM Fit etc.)
        sr_wrapper = None

        if scoring_type == "GMM":
            if smlm_coords.shape[0] < 2:
                return None, 0
            # GMM Fitting ist Teil der Setup-Zeit
            _, gmm_sel_obj, gmm_sel_mean, gmm_sel_cov, gmm_sel_weight = test_gmm_components(smlm_coords)

            sr_wrapper = ScoringRestraintWrapper(
                self.m, self.avs,
                gmm_sel_components=gmm_sel_obj.n_components,
                gmm_sel_mean=gmm_sel_mean,
                gmm_sel_cov=gmm_sel_cov,
                gmm_sel_weight=gmm_sel_weight,
                type="GMM"
            )

        elif scoring_type == "Tree":
            sr_wrapper = ScoringRestraintWrapper(
                self.m, self.avs,
                kdtree_obj=kdtree_obj,
                dataxyz=smlm_coords_tree,
                var=smlm_vars,
                type="Tree"
            )

        elif scoring_type == "Distance":
            smlm_cov_list = []
            for var_scalar in smlm_vars:
                safe_var = max(var_scalar, 1e-9)
                smlm_cov_list.append(np.eye(3) * safe_var)

            sr_wrapper = ScoringRestraintWrapper(
                self.m, self.avs,
                dataxyz=smlm_coords,
                var=smlm_cov_list,
                type="Distance"
            )

        # IMP Scoring Function erstellen (um Overhead zu prüfen)
        sf = IMP.core.RestraintsScoringFunction(sr_wrapper.rs)

        end_time = time.perf_counter()
        setup_time = end_time - start_time

        return sr_wrapper, setup_time

    def benchmark_scoring_latency(self, scoring_types: List[str], n_repeats: int = 100):
        """
        Misst die durchschnittliche Zeit für einen Scoring-Schritt (evaluate).
        """
        results = {}
        print(f"Running Latency Benchmark ({n_repeats} repeats)...")

        for stype in scoring_types:
            print(f"  Testing {stype}...")
            try:
                wrapper, setup_time = self._prepare_scorer(stype)
                if wrapper is None:
                    print(f"    Skipping {stype} (insufficient data)")
                    continue

                # Warmup
                wrapper.evaluate()

                # Measurement
                times = []
                for _ in range(n_repeats):
                    t0 = time.perf_counter()
                    score = wrapper.evaluate()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                avg_time = np.mean(times)
                std_time = np.std(times)
                results[stype] = {
                    "setup_time_sec": setup_time,
                    "avg_eval_time_sec": avg_time,
                    "std_eval_time_sec": std_time,
                    "evals_per_second": 1.0 / avg_time if avg_time > 0 else 0
                }
            except Exception as e:
                print(f"    Error benchmarking {stype}: {e}")

        return pd.DataFrame(results).T

    def benchmark_data_scaling(self, scoring_types: List[str], fractions: List[float]):
        """
        Misst die Scoring-Zeit in Abhängigkeit von der Datenmenge.
        """
        scaling_results = []
        print("Running Data Scaling Benchmark...")

        for frac in fractions:
            n_points = int(len(self.raw_smlm_data) * frac)
            print(f"  Testing fraction {frac} (~{n_points} points)...")

            for stype in scoring_types:
                try:
                    wrapper, _ = self._prepare_scorer(stype, data_subset_fraction=frac)
                    if wrapper is None: continue

                    # Measure single evaluation time (average of 10)
                    t_start = time.perf_counter()
                    for _ in range(10):
                        wrapper.evaluate()
                    avg_time = (time.perf_counter() - t_start) / 10.0

                    scaling_results.append({
                        "scoring_type": stype,
                        "fraction": frac,
                        "n_points": n_points,
                        "avg_time_sec": avg_time
                    })
                except Exception as e:
                    print(f"    Error {stype} at {frac}: {e}")

        return pd.DataFrame(scaling_results)


def main():
    # Instanzieren
    bm = ScoringBenchmark()

    types_to_test = ["Distance", "Tree", "GMM"]  # Wu et al., Tree, Bonomi et al.

    # 1. Latency & Setup Benchmark
    print("\n=== 1. Setup & Latency Benchmark ===")
    latency_df = bm.benchmark_scoring_latency(types_to_test, n_repeats=50)
    print(latency_df)
    latency_df.to_csv("benchmark_latency_results.csv")

    # 2. Scaling Benchmark
    print("\n=== 2. Data Scaling Benchmark ===")
    # Teste mit 10%, 20%, ... 100% der Daten
    fractions = [0.1, 0.3, 0.5, 0.8, 1.0]
    scaling_df = bm.benchmark_data_scaling(types_to_test, fractions)
    print(scaling_df)
    scaling_df.to_csv("benchmark_scaling_results.csv")

    # Optional: Einfacher Plot
    try:
        for stype in types_to_test:
            subset = scaling_df[scaling_df["scoring_type"] == stype]
            if not subset.empty:
                plt.plot(subset["n_points"], subset["avg_time_sec"], marker='o', label=stype)

        plt.xlabel("Number of Data Points")
        plt.ylabel("Time per Evaluation (s)")
        plt.title("SMLM Scoring Efficiency Scaling")
        plt.legend()
        plt.grid(True)
        plt.savefig("benchmark_scaling_plot.png")
        print("\nPlot saved to benchmark_scaling_plot.png")
    except Exception as e:
        print(f"Could not create plot: {e}")


if __name__ == "__main__":
    main()

