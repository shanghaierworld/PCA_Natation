# Script for PCA analysis
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from IPython.display import display
from matplotlib import pyplot as plt


def perform_pca(data):

    """
    Perform PCA on the dataset and provide variance explained details.

    Parameters:
    - data: DataFrame or array-like, the dataset to perform PCA on.

    Returns:
    - explained_variance_ratio: Variance explained by each principal component.
    - total_variance_explained: Total variance explained by all components.
    - suggested_components: Suggested number of components to retain (at least 2).
    - df_pca: DataFrame containing the principal components and clusters.
    """
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data)

    pca = PCA()
    scores = pca.fit_transform(X_std)
    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance_explained = explained_variance_ratio.sum()

    # Ensure at least 2 components are suggested, or more if variance explained exceeds threshold
    cumulative_variance = explained_variance_ratio.cumsum()
    suggested_components = max(2, (cumulative_variance >= 0.75).argmax() + 1)

    # Print variance explained by each component in French
    for i, variance in enumerate(explained_variance_ratio):
        print(f"Composante {i+1} : {variance*100:.2f}% de variance expliquée")

    print(f"Variance totale expliquée : {total_variance_explained*100:.2f}%")
    print(f"Nombre de composantes suggérées : {suggested_components}")

    # Create a DataFrame for the selected principal components
    df_pca = pd.DataFrame(scores[:, :suggested_components], columns=[f"PC{i+1}" for i in range(suggested_components)])

    # Reintroduit les noms des variables comme index du df, servant de coefficients de chaque composante.
    df_pca.index = range(1, len(data) + 1)

    # Affichage des clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(scores[:, :2])  # Utilisation des deux premières composantes principales
    df_pca['cluster'] = clusters

    print("Clusters des nageurs :")
    display(df_pca)


    # Plot 1: individuals on the PC1 x PC2 plane
    fig1, ax1 = plt.subplots(figsize=(12,6))
    ax1.scatter(scores[:,0], scores[:,1])
    for i, txt in enumerate(range(1, len(data)+1)):
        ax1.annotate(str(txt), (scores[i,0], scores[i,1]), fontsize=8)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax1.set_title("plan factoriel: individus (nageurs)")
    ax1.axhline(0, linewidth=0.5, color='black', linestyle='--')
    ax1.axvline(0, linewidth=0.5, color='black', linestyle='--')
    plt.show()


    # unit circle

    corr_var_comp = np.corrcoef(X_std.T, scores.T)[:X_std.shape[1], X_std.shape[1]:]

    fig2, ax2 = plt.subplots(figsize=(6,6))
    circle = plt.Circle((0,0), 1, color='black', fill=False, linewidth=0.7)
    ax2.add_artist(circle)
    # arrows for each variable
    for i, var in enumerate(data.columns):
        x = corr_var_comp[i,0]
        y = corr_var_comp[i,1]
        ax2.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, length_includes_head=True)
        ax2.text(x*1.1, y*1.1, var, fontsize=9)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Correlation circle: variables ↔ components")
    ax2.axhline(0, linewidth=0.5, color='black', linestyle='--')
    ax2.axvline(0, linewidth=0.5, color='black', linestyle='--')
    ax2.set_aspect('equal', 'box')
    plt.show()

    # affichage des clusters
    fig3, ax3 = plt.subplots(figsize=(12,6))
    scatter = ax3.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['cluster'], cmap='viridis')
    legend1 = ax3.legend(*scatter.legend_elements(), title="Clusters")
    ax3.add_artist(legend1)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.axhline(0, linewidth=0.5, color='black', linestyle='--')
    ax3.axvline(0, linewidth=0.5, color='black', linestyle='--')
    ax3.set_title("Clusters des nageurs sur les deux premières composantes principales")
    plt.show()

    #detection des nageurs atypiques
    fig4, ax4 = plt.subplots(figsize=(12,6))
    df_pca['distance'] = np.sqrt(df_pca['PC1']**2 + df_pca['PC2']**2)
    outliers = df_pca.sort_values('distance', ascending=False).head(5)
    
    plt.scatter(df_pca['PC1'], df_pca['PC2'])
    plt.scatter(outliers['PC1'], outliers['PC2'], color='red')
    plt.title("Détection des nageurs atypiques")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axhline(0, linewidth=0.5, color='black', linestyle='--')
    plt.axvline(0, linewidth=0.5, color='black', linestyle='--')
    plt.show()

    return explained_variance_ratio, total_variance_explained, suggested_components, df_pca

