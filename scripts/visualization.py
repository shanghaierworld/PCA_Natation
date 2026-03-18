# Script for data visualization
import matplotlib.pyplot as plt

def plot_pca_results(principal_components, variance_ratio):
    """Plot the PCA results."""
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7)
    plt.title("PCA Results")
    plt.xlabel(f"PC1 ({variance_ratio[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({variance_ratio[1]*100:.2f}%)")
    plt.grid()
    plt.show()