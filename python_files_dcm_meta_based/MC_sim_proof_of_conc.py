import numpy as np

def main():

    background = np.empty([10, 10,10], dtype=str)
    background[:] = 'o' #outside of prostate
    background[3:8,3:8,3:8] = "p" #prostate
    background[3:5,3:5,3:5] = "d" #DIL
    print(background)

    biopsy_points_background_coordinates = np.empty([10,3], dtype=float) # 10 samples, 3 coordinates
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
        heading_vec = np.array([bx_centroid_line[1]-bx_centroid_line[0]])
        queried_bx_pt_vec_from_bx_origin = self.BX_pt_bg_coords - bx_origin
        !!project onto heading vector of biopsy!! to determine coordinates in cylindrical


if __name__ == '__main__':    
    main()
        