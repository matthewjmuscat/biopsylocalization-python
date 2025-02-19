from sklearn.decomposition import PCA
import numpy as np
import cupy as cp

def linear_fitter(data):
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    data = np.array(data).T
    pca = PCA(n_components=1)
    pca.fit(data)
    direction_vector = pca.components_
    #print(direction_vector)

    # create line
    origin = np.mean(data, axis=0)
    euclidian_distance = np.linalg.norm(data - origin, axis=1)
    extent = np.max(euclidian_distance)

    line = np.vstack((origin - direction_vector * extent,
                    origin + direction_vector * extent))

    return line