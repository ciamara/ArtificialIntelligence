import numpy as np

def initialize_centroids_forgy(data, k):

    # TODO forgy initialization
    # random k data points as initial centroids
    
    n_samples = data.shape[0] # rows num in dataset

    if k > n_samples:
        raise ValueError("num of clusters k cannot be greater than the num of data points")

    # random k indices
    indices = np.random.choice(len(data), size=k, replace=False)

    return np.array(data[indices])


def initialize_centroids_kmeans_pp(data, k):

    # TODO kmeans++ initizalization
    # spread centroids efficiently

    n_samples, n_features = data.shape

    if k > n_samples:
        raise ValueError("num of clusters k cannot be greater than the num of data points")

    # first centroid random
    centroids = []
    first_index = np.random.choice(n_samples)
    centroids.append(data[first_index])

    # remaining n-1 centroids
    for _ in range(1, k):
        # distances from each point to the nearest already chosen centroid (euclidean distance)
        distances = np.array([
            min(np.linalg.norm(point - centroid)**2 for centroid in centroids)
            for point in data
        ])

        # choose point with biggest distance
        next_index = np.argmax(distances)
        centroids.append(data[next_index])

    return np.array(centroids)


def assign_to_cluster(data, centroid):

    # TODO find the closest cluster for each data point
    # assigns data points to nearest centroid

    # calculate distances between each data point and centroid (euclidean)
    distances = np.linalg.norm(data[:, np.newaxis] - centroid, axis=2)
    # chooses closest centroid
    assignments = np.argmin(distances, axis=1)
    return assignments
    

def update_centroids(data, assignments):
     
    # TODO find new centroids based on the assignments
    

    k = np.max(assignments) + 1

    # for each cluster selects its data points -> computes mean -> mean=new centroid
    centroids = np.array([
        data[assignments == i].mean(axis=0) if np.any(assignments == i) else np.zeros(data.shape[1])
        for i in range(k)
    ])

    return np.array(centroids)


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

