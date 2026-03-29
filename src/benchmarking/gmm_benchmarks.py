"""Benchmarking utilities for GMM component selection."""

import time
import numpy as np
import matplotlib.pyplot as plt
from smlm_score.src.imp_modeling.scoring.gmm_score import test_gmm_components


class gmm_benchmarking():
    """Benchmarking suite for GMM component fitting performance."""

    @staticmethod
    def test_gmm_components_with_timing(data, component_max=1000):
        """
        Run GMM component selection and measure wall-clock time.

        Parameters
        ----------
        data : np.ndarray
            Input point cloud, shape (N, D).
        component_max : int
            Maximum number of GMM components to test.

        Returns
        -------
        result : dict
            Result dictionary from ``test_gmm_components``.
        times : np.ndarray
            Linearly interpolated time points per iteration.
        """
        start_time = time.time()
        result = test_gmm_components(data, component_max=component_max)
        total_time = time.time() - start_time

        n_iterations = len(result['n_components'])
        times = np.linspace(0, total_time, n_iterations)

        return result, times

    @staticmethod
    def plot_gmmComponent_benchmark(results):
        """
        Plot GMM benchmarking results: execution time and BIC vs. components.

        Parameters
        ----------
        results : list of dict
            Each dict must contain 'n_components', 'data_size', 'times', 'bic'.
        """
        fig = plt.figure(figsize=(20, 15))

        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        for result in results:
            ax1.plot(result['n_components'], [result['data_size']] * len(result['n_components']), result['times'])
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Data Size')
        ax1.set_zlabel('Execution Time (s)')
        ax1.set_title('3D View: Components vs Data Size vs Time')

        # Time vs Components
        ax2 = fig.add_subplot(222)
        for result in results:
            ax2.plot(result['n_components'], result['times'], label='Data size: {}'.format(result['data_size']))
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Time vs Components for Each Dataset')
        ax2.legend()
        ax2.grid(True)

        # Time vs Data Size
        ax3 = fig.add_subplot(223)
        total_times = [result['times'][-1] for result in results]
        data_sizes = [result['data_size'] for result in results]
        ax3.plot(data_sizes, total_times, 'bo-')
        ax3.set_xlabel('Data Size')
        ax3.set_ylabel('Total Execution Time (s)')
        ax3.set_title('Total Execution Time vs Data Size')
        ax3.grid(True)

        # BIC vs Components
        ax4 = fig.add_subplot(224)
        for result in results:
            ax4.plot(result['n_components'], result['bic'], label='Data size: {}'.format(result['data_size']))
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('BIC')
        ax4.set_title('BIC vs Components for Each Dataset')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()