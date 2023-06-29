import numpy as np

def centroid_finder_norm_based():
    print('hello')


def centroid_finder_mean_based(points):
    sumx = sum(points[0])
    sumy = sum(points[1])
    centroid_non_normal = [sumx,sumy]
    centroid = [x*1/len(points[0]) for x in centroid_non_normal]
    return centroid

def centroid_finder_mean_based_numpy(points):
    centroid = np.mean(points,axis=0)
    return centroid


def centeroidfinder_numpy_3D(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return np.array([[sum_x/length, sum_y/length, sum_z/length]])





def main():
    print('This is the centroid finder programme, it is not meant to be run directly.')

if __name__ == '__main__':    
    main()