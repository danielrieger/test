import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from typing import List, Dict
from sklearn import mixture

# Import project modules (Analog zu NPC_example_BD.py)
from smlm_score.src.utility.input import read_experimental_data
from smlm_score.src.utility.data_handling import flexible_filter_smlm_data


class GMMComponentBenchmark:
    """
    Benchmark-Klasse zur Messung der Laufzeit für die Bestimmung der optimalen
    GMM-Komponentenanzahl bei unterschiedlichen Datengrößen.
    """

    def __init__(self, smlm_data_path: str = "ShareLoc_Data/data.csv"):
        self.smlm_data_path = smlm_data_path
        self.raw_smlm_data = None
        self._load_data()

    def _load_data(self):
        """Lädt die SMLM-Rohdaten."""
        print(f"Loading SMLM Data from {self.smlm_data_path}...")
        self.raw_smlm_data = read_experimental_data(self.smlm_data_path)
        if self.raw_smlm_data is None:
            raise ValueError("Failed to load SMLM data.")
        print(f"Data loaded. Total rows: {len(self.raw_smlm_data)}")

    def _get_data_subset(self, fraction: float) -> np.ndarray:
        """
        Erstellt ein Subset der Daten und filtert es (analog zum Hauptskript).
        Gibt das numpy array der Koordinaten zurück.
        """
        # Sample data
        if fraction >= 1.0:
            current_df = self.raw_smlm_data
        else:
            current_df = self.raw_smlm_data.sample(frac=fraction, random_state=42)

        # Filtern (Standard Cut aus NPC_example)
        smlm_coords, _, _, _, _ = flexible_filter_smlm_data(
            current_df,
            filter_type='cut',
            x_cut=(10000, 12000),
            y_cut=(0, 5000),
            fill_z_value=0.0,
            return_tree=False
        )
        return smlm_coords

    def run_component_test(self, data: np.ndarray, component_max: int = 1024) -> float:
        """
        Führt den GMM-Fit-Test durch (test_gmm_components Logik) und misst die Zeit.
        Gibt die Zeit in Sekunden zurück.
        """
        start_time = time.perf_counter()

        # Logik analog zu gmm_score.py -> test_gmm_components
        # Wir replizieren hier die Logik, um nur die Zeit zu messen, ohne Overhead
        n_components = []
        i = 1
        while i <= component_max:
            n_components.append(i)
            i *= 2

        bics = []
        # Wir nutzen tqdm hier nicht für die Output-Sauberkeit im Benchmark
        for n_comp in n_components:
            clf = mixture.GaussianMixture(n_components=n_comp, covariance_type="full")
            clf.fit(data)
            bics.append(clf.bic(data))

        # n = np.argmin(bics) # Würde das Optimum wählen

        end_time = time.perf_counter()
        return end_time - start_time

    def run_benchmark(self, fractions: List[float]):
        """Führt das Benchmark für alle Fractions durch."""
        results = []

        print(f"\n--- Starting GMM Component Benchmark ---")
        print(f"Fractions to test: {fractions}")

        for frac in fractions:
            try:
                # 1. Daten vorbereiten
                data = self._get_data_subset(frac)
                n_points = len(data)

                if n_points < 10:
                    print(f"Skipping fraction {frac} (too few points: {n_points})")
                    continue

                print(f"Testing fraction {frac} ({n_points} points)...")

                # 2. Zeit messen
                duration = self.run_component_test(data)

                print(f"  -> Duration: {duration:.4f} s")

                results.append({
                    "fraction": frac,
                    "n_points": n_points,
                    "duration_sec": duration
                })

            except Exception as e:
                print(f"Error at fraction {frac}: {e}")

        return pd.DataFrame(results)

    def plot_results(self, df: pd.DataFrame):
        """Plottet die Ergebnisse."""
        if df.empty:
            print("No results to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(df["n_points"], df["duration_sec"], marker='o', linestyle='-', linewidth=2, color='purple',
                 label='GMM Component Search')

        plt.title("GMM Component Determination Time vs. Data Size", fontsize=14)
        plt.xlabel("Number of Data Points (N)", fontsize=12)
        plt.ylabel("Time (s)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Speichern und Anzeigen
        plt.savefig("gmm_component_benchmark_plot.png", dpi=300)
        # plt.show() # Optional, falls interaktiv gewünscht
        print("\nPlot saved to 'gmm_component_benchmark_plot.png'")


def main():
    # 1. Benchmark initialisieren
    bm = GMMComponentBenchmark()

    # 2. Fractions definieren (10% bis 100%)
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    # 3. Ausführen
    results_df = bm.run_benchmark(fractions)

    # 4. Speichern
    results_df.to_csv("gmm_component_benchmark_results.csv", index=False)
    print("\nResults saved to 'gmm_component_benchmark_results.csv'")

    # 5. Plotten
    bm.plot_results(results_df)


if __name__ == "__main__":
    main()
