import time
import biopsy_creator
import loading_tools # imported for more sophisticated loading bar
import numpy as np
import open3d as o3d


def simulator(master_structure_reference_dict, structs_referenced_list, num_simulations):

    ref_list = ["Bx ref","OAR ref","DIL ref"] # note that Bx ref has to be the first entry for other parts of the code to work!
    uncertainty_sources_list = []
    uncertainty_sources_raw_dict = {}

    num_biopsies = sum([len(patient_dict[1][ref_list[0]]) for patient_dict in master_structure_reference_dict.items()])

    print('Number of biopsy tracks to simulate are: ', num_biopsies,'.')
    print('Number of simulations per biopsy track has been set to: ', num_simulations,'.')
    

    st = time.time()
    with loading_tools.Loader(num_biopsies,"Reconstructing and sampling biopsies...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            Bx_structs = structs_referenced_list[0]
            for specific_BX_structure_index, specific_BX_structure in enumerate(pydicom_item[Bx_structs]):
                centroid_line = specific_BX_structure["Best fit line of centroid pts"]
                origin_to_first_centroid_vector = specific_BX_structure["Centroid line sample pts"][0]
                list_origin_to_first_centroid_vector = np.squeeze(origin_to_first_centroid_vector).tolist()
                biopsy_samples = biopsy_creator.biopsy_points_reconstruction_and_uniform_sampler(list_origin_to_first_centroid_vector,centroid_line)
                biopsy_samples_point_cloud = o3d.geometry.PointCloud()
                biopsy_samples_point_cloud.points = o3d.utility.Vector3dVector(biopsy_samples[:,0:3])
                pcd_color = np.random.uniform(0, 0.7, size=3)
                biopsy_samples_point_cloud.paint_uniform_color(pcd_color)
                master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Random uniformly sampled volume pts"] = biopsy_samples
                biopsy_raw_point_cloud = master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Point cloud"]
                pcd_color = np.random.uniform(0, 0.7, size=3)
                biopsy_raw_point_cloud.paint_uniform_color(pcd_color)

                # plot point clouds?
                #o3d.visualization.draw_geometries([biopsy_raw_point_cloud,biopsy_samples_point_cloud])



    with loading_tools.Loader(num_biopsies*num_simulations,"Simulating biopsy uncertainties...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            Bx_structs = structs_referenced_list[0]
            for specific_BX_structure_index, specific_BX_structure in enumerate(pydicom_item[Bx_structs]):
                structure_uncertainty_array = np.empty([num_simulations, specific_BX_structure["Uncertainty params"].size])
                for mu,sigma in specific_BX_structure["Uncertainty params"]: 
                    specific_uncertainty_array = np.random.normal(mu, sigma, num_simulations)

                    for j in range(0, num_simulations):
                        print(1)
                        
    return master_structure_reference_dict


    