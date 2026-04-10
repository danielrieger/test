import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkPlotter:
    """
    Klasse zum Erstellen verschiedener Plot-Varianten aus gespeicherten Benchmark-Daten.
    Nutzt vorhandene CSV-Dateien - kein erneutes Ausführen der Benchmarks nötig.
    """

    def __init__(self,
                 latency_csv: str = "benchmark_latency_results.csv",
                 scaling_csv: str = "benchmark_scaling_results.csv"):
        """Lädt die CSV-Dateien."""
        try:
            self.latency_df = pd.read_csv(latency_csv, index_col=0)
            print(f"Loaded {latency_csv}")
        except FileNotFoundError:
            print(f"Warning: {latency_csv} not found.")
            self.latency_df = None

        try:
            self.scaling_df = pd.read_csv(scaling_csv)
            print(f"Loaded {scaling_csv}")
        except FileNotFoundError:
            print(f"Warning: {scaling_csv} not found.")
            self.scaling_df = None

    def plot_scaling_linear(self, filename="plot_scaling_linear.png"):
        """Standard linearer Plot (Original)."""
        if self.scaling_df is None:
            print("No scaling data available.")
            return

        plt.figure(figsize=(10, 6))
        for stype in self.scaling_df["scoring_type"].unique():
            subset = self.scaling_df[self.scaling_df["scoring_type"] == stype]
            plt.plot(subset["n_points"], subset["avg_time_sec"],
                     marker='o', linewidth=2, markersize=8, label=stype)

        plt.xlabel("Number of Data Points", fontsize=12)
        plt.ylabel("Time per Evaluation (s)", fontsize=12)
        plt.title("SMLM Scoring Efficiency - Linear Scale", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved: {filename}")

    def plot_scaling_log(self, filename="plot_scaling_log.png"):
        """Logarithmische Y-Achse - bessere Darstellung bei großen Unterschieden."""
        if self.scaling_df is None:
            print("No scaling data available.")
            return

        plt.figure(figsize=(10, 6))
        for stype in self.scaling_df["scoring_type"].unique():
            subset = self.scaling_df[self.scaling_df["scoring_type"] == stype]
            plt.semilogy(subset["n_points"], subset["avg_time_sec"],
                         marker='o', linewidth=2, markersize=8, label=stype)

        plt.xlabel("Number of Data Points", fontsize=12)
        plt.ylabel("Time per Evaluation (s, log scale)", fontsize=12)
        plt.title("SMLM Scoring Efficiency - Logarithmic Scale", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved: {filename}")

    def plot_scaling_subplots(self, filename="plot_scaling_subplots.png"):
        """Separate Subplots für jede Methode - ideal bei extremen Unterschieden."""
        if self.scaling_df is None:
            print("No scaling data available.")
            return

        scoring_types = self.scaling_df["scoring_type"].unique()
        n_plots = len(scoring_types)

        fig, axes = plt.subplots(1, n_plots, figsize=(15, 5), sharey=False)
        if n_plots == 1:
            axes = [axes]

        for idx, stype in enumerate(scoring_types):
            subset = self.scaling_df[self.scaling_df["scoring_type"] == stype]
            axes[idx].plot(subset["n_points"], subset["avg_time_sec"],
                           marker='o', linewidth=2, markersize=8, color=f"C{idx}")
            axes[idx].set_xlabel("Data Points", fontsize=11)
            axes[idx].set_ylabel("Time (s)", fontsize=11)
            axes[idx].set_title(f"{stype} Scoring", fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

        fig.suptitle("SMLM Scoring Efficiency - Individual Scales",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved: {filename}")

    def plot_scaling_normalized(self, filename="plot_scaling_normalized.png"):
        """Normalisiert auf den schnellsten Wert jeder Methode (relative Performance)."""
        if self.scaling_df is None:
            print("No scaling data available.")
            return

        plt.figure(figsize=(10, 6))
        for stype in self.scaling_df["scoring_type"].unique():
            subset = self.scaling_df[self.scaling_df["scoring_type"] == stype].copy()
            # Normalisiere auf Minimum (= 1.0)
            min_time = subset["avg_time_sec"].min()
            subset["relative_time"] = subset["avg_time_sec"] / min_time

            plt.plot(subset["n_points"], subset["relative_time"],
                     marker='o', linewidth=2, markersize=8, label=stype)

        plt.xlabel("Number of Data Points", fontsize=12)
        plt.ylabel("Relative Time (normalized to fastest)", fontsize=12)
        plt.title("SMLM Scoring - Relative Scaling Behavior", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved: {filename}")

    def plot_latency_comparison(self, filename="plot_latency_comparison.png"):
        """Balkendiagramm für Setup- und Evaluationszeiten."""
        if self.latency_df is None:
            print("No latency data available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Setup Times
        setup_times = self.latency_df["setup_time_sec"]
        ax1.barh(setup_times.index, setup_times.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_xlabel("Setup Time (s)", fontsize=12)
        ax1.set_title("Initialization Overhead", fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Evaluation Times (log scale wegen großer Unterschiede)
        eval_times = self.latency_df["avg_eval_time_sec"]
        ax2.barh(eval_times.index, eval_times.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_xlabel("Avg. Evaluation Time (s, log scale)", fontsize=12)
        ax2.set_xscale('log')
        ax2.set_title("Scoring Latency (per call)", fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x', which='both')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved: {filename}")

    def plot_speedup_factor(self, baseline="Distance", filename="plot_speedup_factor.png"):
        """Zeigt Speedup-Faktor relativ zu einer Baseline-Methode."""
        if self.latency_df is None:
            print("No latency data available.")
            return

        if baseline not in self.latency_df.index:
            print(f"Baseline '{baseline}' not found. Available: {self.latency_df.index.tolist()}")
            return

        baseline_time = self.latency_df.loc[baseline, "avg_eval_time_sec"]
        speedup = baseline_time / self.latency_df["avg_eval_time_sec"]

        plt.figure(figsize=(10, 6))
        colors = ['gray' if idx == baseline else f'C{i}'
                  for i, idx in enumerate(speedup.index)]
        bars = plt.bar(speedup.index, speedup.values, color=colors, edgecolor='black', linewidth=1.5)

        plt.ylabel(f"Speedup Factor vs. {baseline}", fontsize=12)
        plt.title(f"Scoring Performance: Speedup Relative to {baseline}",
                  fontsize=14, fontweight='bold')
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11)

        # Werte auf Balken schreiben
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved: {filename}")

    def create_all_plots(self):
        """Erstellt alle Plot-Varianten auf einmal."""
        print("\n=== Creating All Plot Variants ===\n")
        self.plot_scaling_linear()
        self.plot_scaling_log()
        self.plot_scaling_subplots()
        self.plot_scaling_normalized()
        self.plot_latency_comparison()
        self.plot_speedup_factor()
        print("\n=== All plots created! ===")


# VERWENDUNG
if __name__ == "__main__":
    plotter = BenchmarkPlotter()

    # Option 1: Alle Plots auf einmal
    plotter.create_all_plots()

    # Option 2: Einzelne Plots (auskommentieren nach Bedarf)
    # plotter.plot_scaling_log()
    # plotter.plot_scaling_subplots()
    # plotter.plot_speedup_factor(baseline="Tree")  # oder "GMM"
