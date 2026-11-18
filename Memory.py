import os.path
from collections import defaultdict
import torch
import numpy as np

from sklearn.cluster import KMeans
from clustering import *


class Memory:
    def __init__(self):
        self.memory = defaultdict(self.default_values)
        self.average_std = None

    def default_values(self):
        return {'embeddings': [], 'num_old_embeddings': 0, 'mean_embedding': None, 'std_embedding': None,
                'centroids': None, 'stds': None, 'num_vectors': None, 'indices': None}


    def clear_record(self, class_name):
        for key, record in self.memory.items():
            if key == class_name:
                print(f'clearing memory for class: {class_name}')
                self.memory[key]['embeddings'] = []
                self.memory[key]['num_old_embeddings'] = 0
                self.memory[key]['mean_embedding'] = None
                self.memory[key]['std_embedding'] = None
                self.memory[key]['centroids'] = None
                self.memory[key]['stds'] = None
                self.memory[key]['num_vectors'] = None
                self.memory[key]['indices'] = None
                self.memory[key]['is_familiar'] = False

                return


    def add_embedding_to_class(self, embedding, class_name, is_familiar = False, closest_cent = None, update_memory = False,
                               clustering_radius=0.3, clustering_distance_threshold=0.3):
        if class_name not in self.memory.keys():
            print(f'creating new memory entry for class_name: {class_name}')
            self.memory[class_name] = self.default_values()
            self.memory[class_name]['is_familiar'] = is_familiar

        self.memory[class_name]['embeddings'].append(embedding)
        if closest_cent is not None:
            # we want to update the centroid of this class, based on the new image
            # but we only want to update the std, not the centroid
            class_data = self.memory[class_name]
            old_mean = class_data['centroids'][closest_cent]
            old_std = class_data['stds'][closest_cent]
            old_count = class_data['num_vectors'][closest_cent]

            new_mean, new_std, new_num_vectors = (
                self.update_mean_and_std(old_mean, old_std, old_count, np.array([embedding])))

            # class_data['centroids'][closest_cent] = new_mean
            class_data['stds'][closest_cent] = new_std
            class_data['num_vectors'][closest_cent] = new_num_vectors
        if update_memory:
            self.calc_mean_std(clustering_radius, clustering_distance_threshold)


    def calc_mean_std(self, clustering_radius, clustering_distance_threshold):
        # Calculate mean and std after collecting all the data
        for class_name, value in self.memory.items():
            embeddings_list = value['embeddings']
            if embeddings_list:
                old_mean = value.get('mean_embedding', np.zeros_like(embeddings_list[0]))
                old_std = value.get('std_embedding', np.zeros_like(embeddings_list[0]))
                old_count = value['num_old_embeddings']

                value['mean_embedding'], value['std_embedding'], value['num_old_embeddings'] = (
                    self.update_mean_and_std(old_mean, old_std, old_count, np.array(embeddings_list)))

                labels, cluster_stats = semi_supervised_clustering_with_stats(embeddings_list,
                                                                              known_clusters = {},
                                                                              max_radius=clustering_radius,
                                                                              distance_threshold=clustering_distance_threshold,
                                                                              min_samples=2)
                
                # reduce number of clusters using post-process
                max_num_clusters = 0.1 * len(embeddings_list)
                cluster_stats = merge_clusters_until_limit(cluster_stats,embeddings_list,max_clusters=max_num_clusters)

                value['centroids'] = cluster_stats['centroids']
                value['stds'] = cluster_stats['stds']
                value['num_vectors'] = cluster_stats['sizes']
                value['indices'] = cluster_stats['indices']

                num_clusters = len(value['centroids'])
                print(f'memory calc_mean_std class {class_name} got {num_clusters} clusters')
                # after calculating mean and std we delete the collected embeddings
                value['embeddings'] = []

        self.calc_average_std_vec()


    def calc_average_std_vec(self):
        # go over all classes, collect all stds into one large list
        # then calculate the mean std
        all_arrays = []

        for class_name, value in self.memory.items():
            stds = value['stds']
            for vec in stds.values():
                if not np.all(vec == 0.0): # ignore 0 vectors
                    all_arrays.append(vec)  # Append each ndarray instead of extending

        # Compute the average vector
        if all_arrays:
            self.average_std = np.mean(np.stack(all_arrays), axis=0)
        else:
            self.average_std = None  # Handle the case where there are no vectors


    def update_mean_and_std(self, old_mean, old_std, old_count, new_samples):
        if old_count == 0:
            # there are no old mean and old std, calculate them based on the current samples
            # get the vector that is closest to the mean

            new_mean = np.mean(new_samples, axis=0)
            simple_mean = False
            # new_mean = new_mean / np.linalg.norm(new_mean)
            if simple_mean:
                #print("Memory:update_mean_and_std simple_mean is True")
                new_mean = new_mean
            else:
                #print("Memory:update_mean_and_std simple_mean is False")
                distances_to_mean = np.linalg.norm(new_samples - new_mean, axis=1)
                mean_closest_vector_idx = np.argmin(distances_to_mean)
                new_mean = new_samples[mean_closest_vector_idx]

            new_std = np.std(new_samples, axis=0)
            old_count = len(new_samples)
            return new_mean, new_std, old_count

        else:
            # calculate the new mean and std based also on old values
            new_count = len(new_samples)
            if new_count == 0:
                return old_mean, old_std  # No new samples to update

            sum_squared_diff = np.sum((new_samples - old_mean) ** 2)

            new_mean = (old_count * old_mean + np.sum(new_samples)) / (old_count + new_count)
            new_variance = ((old_count - 1) * (old_std ** 2) + sum_squared_diff +
                            (new_count / (old_count + new_count)) * (old_mean - new_mean) ** 2) / (
                                   old_count + new_count - 1)

            new_std = np.sqrt(new_variance)

            return new_mean, new_std, old_count + new_count

    def cluster_and_store_stats(self, vectors, k):

        # Convert the input list to a numpy array
        data = np.array(vectors)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        # Get centroids
        centroids = kmeans.cluster_centers_

        # Compute standard deviations for each cluster and number of vectors per cluster
        stds = []
        num_vectors = []
        for i in range(k):
            cluster_points = data[kmeans.labels_ == i]
            cluster_std = np.std(cluster_points, axis=0) if len(cluster_points) > 0 else np.zeros(data.shape[1])
            stds.append(cluster_std)
            num_vectors.append(len(cluster_points))

        return centroids, stds, num_vectors

    def reconstruct_from_centroids_and_stds(self):

        # Generate samples around the centroid
        generated_samples = {}

        for class_name, value in self.memory.items():
            class_vectors = []
            for centroid, std, num_samples in zip(value['centroids'].values(), value['stds'].values(), value['num_vectors'].values()):
                reconstructed_vectors = np.random.normal(loc=centroid, scale=std*0.1, size=(num_samples, len(centroid)))

                # Ensure that the centroid itself is included in the reconstructed vectors
                reconstructed_vectors[0] = centroid
                class_vectors.append(reconstructed_vectors)

            generated_samples[class_name] = np.concatenate(class_vectors,axis=0)

        return generated_samples

    def get_predicted_class(self, embedding):
        min_dist = float('inf')
        pred_class = None
        closest_cent = None
        for key, val in self.memory.items():
            for cent_key, std in zip(val['centroids'].keys(), val['stds'].values()):
                cent = val['centroids'][cent_key]
                # dist = np.linalg.norm(embedding - cent)
                if np.all(std == 0.0):
                    std = self.average_std
                dist = np.linalg.norm((embedding - cent)/(std))
                if dist < min_dist:
                    pred_class = key
                    min_dist = dist
                    closest_cent = cent_key
        return pred_class, min_dist, closest_cent

    def generate_samples(self, num_samples):
        generated_samples = {}

        for class_name, values in self.memory.items():
            mean_vector = values['mean_embedding']
            std_vector = values['std_embedding']

            if mean_vector is not None and std_vector is not None:
                # test:
                test_scale = std_vector * 0.3
                # test_scale = std_vector
                num_dimensions = len(mean_vector)
                samples = np.random.normal(loc=mean_vector, scale=test_scale, size=(num_samples, num_dimensions))
                samples[0] = mean_vector
                generated_samples[class_name] = samples

        return generated_samples

    def save_memory(self, save_path):
        torch.save({'memory': self.memory, 'average_std': self.average_std}, save_path)

    def load_memory(self, load_path):
        if os.path.exists(load_path):
            loaded_data = torch.load(load_path, weights_only=False)
            self.memory = loaded_data.get('memory', {})
            self.average_std = loaded_data.get('average_std', None)
