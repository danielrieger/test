import matplotlib.pyplot as plt


def make_gmm_component_plot(r: dict) -> None:
    """
    Plot GMM model selection diagnostics (score, AIC, BIC) vs. number of components.

    Parameters
    ----------
    r : dict
        Result dictionary from ``test_gmm_components`` containing keys:
        'n_components', 'score', 'aic', 'bic', 'n' (optimal index).
    """
    n_components = r['n_components']

    plt.figure(figsize=(15, 5))

    # Score plot
    plt.subplot(1, 3, 1)
    plt.plot(n_components, r['score'], marker='o')
    plt.axvline(x=r['n_components'][r['n']], color='red', linestyle='--')
    plt.title('GMM Scores by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Log-Likelihood')
    plt.grid(True)

    # AIC plot
    plt.subplot(1, 3, 2)
    plt.semilogy(n_components, r['aic'], marker='o', color='red')
    plt.axvline(x=r['n_components'][r['n']], color='red', linestyle='--')
    plt.title('GMM AIC by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC')
    plt.grid(True)

    # BIC plot
    plt.subplot(1, 3, 3)
    plt.semilogy(n_components, r['bic'], marker='o', color='green')
    plt.axvline(x=r['n_components'][r['n']], color='red', linestyle='--')
    plt.title('GMM BIC by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.grid(True)

    plt.tight_layout()
    plt.show()