import numpy as np
import math

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import MC_toy_model_funcs


def main():

    background = np.empty([10, 10,10], dtype=str)
    background[:] = 'o' #outside of prostate
    background[3:8,3:8,3:8] = "p" #prostate
    background[3:5,3:5,3:5] = "d" #DIL
    print(background)

    biopsy_samples_num_per_z_per_radii = 20
    biopsy_offset_x = 2
    biopsy_offset_y = 3
    biopsy_offset_z = -4
    biopsy_radius = 2
    biopsy_length = 5
    num_radii = 4
    biopsy_samples_num_per_z = biopsy_samples_num_per_z_per_radii*num_radii

    num_points_per_radii = np.empty(num_radii,dtype=int)
    for i in range(1,num_radii+1,1):
        num_points_per_radii[i-1] = MC_toy_model_funcs.num_points_per_radii(i)

    #biopsy_points_background_coordinates = np.array([biopsy_offset_x,biopsy_offset_y,biopsy_offset_z], dtype=float) 
    biopsy_points_background_coordinates = np.empty([np.sum(num_points_per_radii)*biopsy_length,3], dtype=float) # 20 samples, 3 coordinates

    for z in range(biopsy_length):
        biopsy_points_background_coordinates[0+num_points_per_radii[z]:z*biopsy_samples_num_per_z+biopsy_samples_num_per_z,2] = biopsy_offset_z + z
        for r_iter in range(1,num_radii+1):
            for i in range(biopsy_samples_num_per_z_per_radii):
                biopsy_points_background_coordinates[z*biopsy_samples_num_per_z+biopsy_samples_num_per_z_per_radii*(r_iter-1)+i,0] = biopsy_offset_x + (r_iter/num_radii)*biopsy_radius*math.cos(2*math.pi*i/biopsy_samples_num_per_z_per_radii)
                biopsy_points_background_coordinates[z*biopsy_samples_num_per_z+biopsy_samples_num_per_z_per_radii*(r_iter-1)+i,1] = biopsy_offset_y + (r_iter/num_radii)*biopsy_radius*math.sin(2*math.pi*i/biopsy_samples_num_per_z_per_radii)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(biopsy_points_background_coordinates[:,0], biopsy_points_background_coordinates[:,1], biopsy_points_background_coordinates[:,2], c=biopsy_points_background_coordinates[:,2], cmap='Greens');
    ax.set_xlim(biopsy_offset_x-biopsy_radius-1, biopsy_offset_x+biopsy_radius+1)
    ax.set_ylim(biopsy_offset_y-biopsy_radius-1, biopsy_offset_y+biopsy_radius+1)
    ax.set_zlim(biopsy_offset_z-1, biopsy_offset_z+biopsy_length+1)

    fig.show()

    biopsy_points_biopsy_coordinates = np.empty([10,3], dtype=bool) # 10 samples, 3 coordinates
    num_bx_samples = np.size(biopsy_points_background_coordinates,0)
    biopsy_points = [biopsy_pt(biopsy_points_background_coordinates[x]) for x in range(0,num_bx_samples)]
    print(biopsy_points[0])


class biopsy_pt:
    def __init__(self, queried_BX_pt):
        self.BX_pt_bg_coords = queried_BX_pt
        self.BX_pt_bx_coords = np.empty([1,3], dtype=float)
    def __str__(self):
        return f"{self.BX_pt_bg_coords}"
    def localize_in_bx_coords(self,bx_centroid_line):
        bx_origin = bx_centroid_line[0]
        bx_end = bx_centroid_line[1]
        heading_vec = np.array([bx_end-bx_origin])
        queried_bx_pt_bx_origin = self.BX_pt_bg_coords - bx_origin
        self.BX_pt_bx_coords = queried_bx_pt_bx_origin
        queried_bx_pt_bx_origin_Z = np.dot(queried_bx_pt_bx_origin,heading_vec)/np.linalg.norm(heading_vec)
        queried_bx_pt_bx_origin_r = math.sqrt(np.linalg.norm(queried_bx_pt_bx_origin)^2-np.linalg.norm(queried_bx_pt_bx_origin_Z)^2)


if __name__ == '__main__':    
    main()
        