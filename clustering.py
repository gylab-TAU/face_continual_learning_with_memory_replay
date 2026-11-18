from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import random


def semi_supervised_clustering_with_stats(embeddings, known_clusters, max_radius=0.75, distance_threshold=0.75, min_samples=2):
    """
    Semi-supervised clustering where unknown embeddings are first matched to known clusters
    (with string IDs) if they are within a given maximum radius. Remaining unknown embeddings
    are clustered into new clusters. Centroids are the vectors closest to the mean of the cluster.

    Args:
        embeddings (list of np.array): List of embeddings.
        known_clusters (dict): A dictionary where the key is the cluster label (string) and
                               the value is a list of indices belonging to that cluster.
        max_radius (float): Maximum radius within which an unknown embedding can be matched
                            to a known cluster.
        distance_threshold (float): Maximum distance between points in a new cluster.
        min_samples (int): Minimum number of samples required to form a cluster.

    Returns:
        labels (list): Final labels after semi-supervised clustering.
        cluster_stats (dict): Cluster statistics, including centroids, stds, sizes, and indices.
    """

    simple_mean = False
    print(f'semi_supervised_clustering_with_stats simple_mean is {simple_mean}')

    n = len(embeddings)
    labels = np.full(n, None, dtype=object)  # Initialize all labels as None (unassigned)
    cluster_stats = {"centroids": {}, "stds": {}, "sizes": {}, "indices": {}}

    # Step 1: Assign known clusters
    for cluster_label, indices in known_clusters.items():
        for idx in indices:
            labels[idx] = cluster_label

    # Step 2: Calculate cluster centroids, stds, and indices for known clusters
    known_means = {}
    for cluster_label, indices in known_clusters.items():
        cluster_embeddings = np.array([embeddings[idx] for idx in indices])
        cluster_mean = np.mean(cluster_embeddings, axis=0)
        known_means[cluster_label] = cluster_mean

        # Find the vector closest to the mean
        if simple_mean:
            #print("semi_supervised_clustering_with_stats simple_mean is True")
            centroid = cluster_mean
        else:
            #print("semi_supervised_clustering_with_stats simple_mean is False")
            distances_to_mean = cosine_distances(cluster_embeddings, [cluster_mean]).squeeze()
            closest_to_mean_idx = np.argmin(distances_to_mean)
            centroid = cluster_embeddings[closest_to_mean_idx]

        # Calculate std
        cluster_std = np.std(cluster_embeddings, axis=0)

        # Update stats
        cluster_stats["centroids"][cluster_label] = centroid
        cluster_stats["stds"][cluster_label] = cluster_std
        cluster_stats["sizes"][cluster_label] = len(cluster_embeddings)
        cluster_stats["indices"][cluster_label] = indices

    # Step 3: Extract embeddings that don't belong to any known cluster (those labeled as None)
    unknown_embeddings = [embeddings[i] for i in range(n) if labels[i] is None]
    unknown_idx_list = [i for i in range(n) if labels[i] is None]

    # Phase 1: Attempt to match unknown embeddings to known clusters
    for unknown_idx, embedding in zip(unknown_idx_list, unknown_embeddings):
        assigned = False
        embedding = np.squeeze(embedding)  # Ensure embedding is 2D
        for cluster_label, mean_embedding in known_means.items():
            mean_embedding = np.squeeze(mean_embedding)  # Ensure mean embedding is 2D
            # Calculate distance between unknown embedding and known cluster mean (centroid)
            dist = cosine_distances([embedding], [mean_embedding])[0][0]

            # If the distance is within the max_radius, assign it to the cluster
            if dist <= max_radius:
                labels[unknown_idx] = cluster_label
                cluster_stats["indices"][cluster_label].append(unknown_idx)
                assigned = True
                break

        if not assigned:
            # If no match found, leave it as unassigned (None)
            continue

    # Phase 2: Cluster the remaining unknown embeddings into new clusters
    remaining_unknown_embeddings = [embeddings[i] for i in range(n) if labels[i] is None]

    if len(remaining_unknown_embeddings) >= min_samples:
        # Convert list to numpy array and ensure it's 2D
        remaining_unknown_embeddings = np.array(remaining_unknown_embeddings)
        if remaining_unknown_embeddings.ndim == 3:
            remaining_unknown_embeddings = remaining_unknown_embeddings.reshape(
                remaining_unknown_embeddings.shape[0], -1
            )

        # Perform Agglomerative Clustering on the remaining unknown embeddings
        clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            metric="cosine",
            linkage="average",
        ).fit(remaining_unknown_embeddings)

        unknown_labels = clustering.labels_

        # Assign new cluster labels as numeric IDs starting from 0 or higher
        current_max_label = max(
            [int(label[1:]) for label in labels if isinstance(label, str) and label.startswith("new_")], default=-1
        ) + 1
        unknown_idx = 0
        for i in range(n):
            if labels[i] is None:
                if unknown_labels[unknown_idx] != -1:  # If it's not noise
                    new_cluster_label = f"new_{current_max_label + unknown_labels[unknown_idx]}"
                    labels[i] = new_cluster_label
                    if new_cluster_label not in cluster_stats["indices"]:
                        cluster_stats["indices"][new_cluster_label] = []
                    cluster_stats["indices"][new_cluster_label].append(i)
                unknown_idx += 1

        # Update cluster stats for new clusters
        for cluster_id in range(len(np.unique(unknown_labels))):
            cluster_points = np.array([remaining_unknown_embeddings[idx] for idx, lbl in enumerate(unknown_labels) if lbl == cluster_id])
            cluster_indices = [unknown_idx_list[idx] for idx, lbl in enumerate(unknown_labels) if lbl == cluster_id]

            if cluster_points.size > 0:
                cluster_mean = np.mean(cluster_points, axis=0)

            if simple_mean:
                #print("semi_supervised_clustering_with_stats simple_mean is True")
                centroid = cluster_mean
            else:
                #print("semi_supervised_clustering_with_stats simple_mean is False")
                distances_to_mean = cosine_distances(cluster_points, [cluster_mean]).squeeze()
                closest_to_mean_idx = np.argmin(distances_to_mean)
                centroid = cluster_points[closest_to_mean_idx]

            # Calculate std
            cluster_std = np.std(cluster_points, axis=0)

            cluster_label = f"new_{current_max_label + cluster_id}"
            cluster_stats["centroids"][cluster_label] = centroid
            cluster_stats["stds"][cluster_label] = cluster_std
            cluster_stats["sizes"][cluster_label] = len(cluster_points)
            cluster_stats["indices"][cluster_label] = cluster_indices
    elif len(remaining_unknown_embeddings) == 1:
        # If only one remaining unknown embedding, assign a new unique cluster label
        single_idx = [i for i in range(n) if labels[i] is None][0]
        current_max_label = max(
            [int(label[1:]) for label in labels if isinstance(label, str) and label.startswith("new_")], default=-1
        ) + 1
        cluster_label = f"new_{current_max_label}"
        labels[single_idx] = cluster_label

        # Update cluster stats for the single embedding
        cluster_stats["centroids"][cluster_label] = embeddings[single_idx]
        cluster_stats["stds"][cluster_label] = np.zeros(len(embeddings[single_idx]))
        cluster_stats["sizes"][cluster_label] = 1
        cluster_stats["indices"][cluster_label] = [single_idx]

    return labels, cluster_stats

def merge_clusters_until_limit(cluster_stats, embeddings, max_clusters):
    """
    Iteratively merge the closest clusters (by cosine distance between centroids)
    until the number of clusters <= max_clusters.
    The centroid of each merged cluster is set to the *real* embedding closest
    to the mean of all embeddings in that merged cluster.
    """
    centroids = cluster_stats["centroids"]
    indices = cluster_stats["indices"]

    while len(centroids) > max_clusters:
        labels = list(centroids.keys())
        centroids_array = np.array([centroids[l] for l in labels])

        # Compute pairwise cosine distances between centroids
        dists = cosine_distances(centroids_array)
        np.fill_diagonal(dists, np.inf)  # ignore self-distances

        # Find closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        label_i, label_j = labels[i], labels[j]

        # Merge their indices
        merged_indices = indices[label_i] + indices[label_j]
        merged_embeddings = np.array([embeddings[idx] for idx in merged_indices])

        # Compute the mean embedding (not stored, just for centroid selection)
        mean_emb = np.mean(merged_embeddings, axis=0)

        # Find the *real* embedding closest to the mean
        distances_to_mean = cosine_distances(merged_embeddings, [mean_emb]).squeeze()
        closest_idx = np.argmin(distances_to_mean)
        centroid = merged_embeddings[closest_idx]

        # Create merged cluster
        new_label = f"merged_{label_i}_{label_j}"
        centroids[new_label] = centroid
        indices[new_label] = merged_indices
        cluster_stats["stds"][new_label] = np.std(merged_embeddings, axis=0)
        cluster_stats["sizes"][new_label] = len(merged_embeddings)

        # Remove old clusters completely
        for old in (label_i, label_j):
            for key in cluster_stats:
                if old in cluster_stats[key]:
                    cluster_stats[key].pop(old, None)

    return cluster_stats
