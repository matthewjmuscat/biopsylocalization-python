import math
import numpy as np

def num_points_per_radii(n,N):
    # n is the number of radii steps, N is the number of points for the first radii
    theta = theta_step(n,N)
    num_points = math.floor(2*np.pi/theta)
    return num_points

def theta_step(n,N):
    # n is the number of radii steps, N is the number of points for the first radii
    theta = (2*np.pi/N)/(2**(n-1))
    return theta