import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid


def tune_kmeans(X, max_clusters=10, random_state=42):
    # Determine the optimal number of clusters
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    # Tune other parameters
    param_grid = {
        'init': ['k-means++', 'random'],
        'max_iter': [200, 300],
        'n_init': [10, 20]
    }

    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(param_grid):
        kmeans = KMeans(n_clusters=optimal_clusters, **params, random_state=random_state)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)

        if score > best_score:
            best_score = score
            best_params = params

    best_params['n_clusters'] = optimal_clusters

    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Best parameters: {best_params}")
    print(f"Best silhouette score: {best_score}")

    return best_params
