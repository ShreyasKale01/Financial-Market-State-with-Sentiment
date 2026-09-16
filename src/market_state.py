"""Market-state representation and clustering utilities."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS


def correlation_to_distance(correlation_matrix):
    """Convert correlations into a distance matrix."""
    correlation_matrix = np.asarray(correlation_matrix)
    distance = np.sqrt(2 * (1 - correlation_matrix))
    np.fill_diagonal(distance, 0.0)
    return distance


def mds_embedding(distance_matrix, n_components=2, random_state=42):
    """Embed a distance matrix into a lower-dimensional representation."""
    model = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
    )
    return model.fit_transform(distance_matrix)


def cluster_market_states(features, n_states=4, random_state=42):
    """Cluster market representations into discrete market states."""
    model = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    labels = model.fit_predict(features)
    return labels, model
