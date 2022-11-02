import math
import numpy as np

def num_points_per_radii(n):
    theta = math.acos((n-1+1/2)/n)
    num_points = math.floor(2*np.pi/theta) + 1
    return num_points