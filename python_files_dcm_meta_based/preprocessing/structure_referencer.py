import numpy as np
import pydicom

import misc_tools
from biopsy_optimizer.v2.live_integration import (
    TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY,
    TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY,
)


def structure_referencer(data_removals_dict_bx,
                        data_removals_dict_prostate,
                        data_removals_dict_dil,
                        data_removals_dict_urethra,
                        data_removals_dict_rectum,
                         structure_dcm_dict,
                         dose_dcm_dict,
                         plan_dcm_dict,
                         US_dcms_dict,
                         MR_T2_dcms_dict,
                         MR_ADC_dcms_dict,
                         OAR_list,
                         DIL_list,
                         Bx_list,
                         st_ref_list,
                         structs_referenced_dict,
                         ds_ref,
                         pln_ref,
                         mr_adc_ref,
                         mr_t2_ref,
                         us_ref,
                         all_ref_key,
                         mr_global_multi_structure_output_dataframe_str,
                         mr_global_by_voxel_multi_structure_output_dataframe_str,
                         bx_sim_locations_dict,
                         rectum_list,
                         urethra_list,
                         interp_inter_slice_dist,
                         interp_intra_slice_dist,
                         simulated_biopsy_fraction_numbers_to_create,
                         fraction_prefixes,
                         important_info,
                         live_display
                         ):
    """
    A function that builds a reference library of the dicom elements passed to it so that
    we can match the ROI name to the contour information, since the contour
    information is referenced to the name by a number.
    """
    master_st_ds_ref_dict = {}
    master_st_ds_info_dict = {}
    master_st_ds_info_global_dict = {"Global": None, "By patient": None}

    global_num_biopsies = 0
    global_num_OAR = 0
    global_num_DIL = 0
    global_total_num_structs = 0
    global_num_cases = 0
    global_unique_patient_names_list = []

    for UID, structure_item_path in structure_dcm_dict.items():
        with pydicom.dcmread(structure_item_path, defer_size = '2 MB') as structure_item:

            filtered_OARs = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in OAR_list)]

            ### Remove unwanted data (prostate)
            if UID in data_removals_dict_prostate.keys():
                for prost_id_to_remove in data_removals_dict_prostate[UID]:
                    for prost in filtered_OARs:
                        if prost.ROIName == prost_id_to_remove:
                            filtered_OARs.remove(prost)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Prostate: {prost_id_to_remove})) ", live_display)

            OAR_ref = [{"ROI":x.ROIName,
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[1],
                        "Raw contour pts zslice list": None,
                        "Raw contour pts": None,
                        "Equal num zslice contour pts": None,
                        "Intra-slice interpolation information": None,
                        "Inter-slice interpolation information": None,
                        "Point cloud raw": None,
                        "Delaunay triangulation global structure": None,
                        "Delaunay triangulation zslice-wise list": None,
                        "Structure centroid pts": None,
                        "Best fit line of centroid pts": None,
                        "Centroid line sample pts": None,
                        "Structure global centroid": None,
                        "Reconstructed structure pts arr": None,
                        "Interpolated structure point cloud dict": None,
                        "Reconstructed structure delaunay global": None,
                        "Maximum pairwise distance": None,
                        "Structure volume": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None,
                        "MC data: Generated normal dist random samples arr": None,
                        "KDtree": None,
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_OARs)]

            filtered_DILs = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in DIL_list)]

            ### Remove unwanted data (dils)
            if UID in data_removals_dict_dil.keys():
                for dil_id_to_remove in data_removals_dict_dil[UID]:
                    for dil in filtered_DILs:
                        if dil.ROIName == dil_id_to_remove:
                            filtered_DILs.remove(dil)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, DIL: {dil_id_to_remove})) ", live_display)

            DIL_ref = [{"ROI":x.ROIName,
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[2],
                        "Raw contour pts zslice list": None,
                        "Raw contour pts": None,
                        "Equal num zslice contour pts": None,
                        "Intra-slice interpolation information": None,
                        "Inter-slice interpolation information": None,
                        "Point cloud raw": None,
                        "Delaunay triangulation global structure": None,
                        "Delaunay triangulation zslice-wise list": None,
                        "Structure centroid pts": None,
                        "Best fit line of centroid pts": None,
                        "Centroid line sample pts": None,
                        "Structure global centroid": None,
                        "Reconstructed structure pts arr": None,
                        "Interpolated structure point cloud dict": None,
                        "Reconstructed structure delaunay global": None,
                        "Maximum pairwise distance": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Structure volume": None,
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None,
                        "MC data: Generated normal dist random samples arr": None,
                        "KDtree": None,
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_DILs)]

            filtered_rectums = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in rectum_list)]

            ### Remove unwanted data (rectum)
            if UID in data_removals_dict_rectum.keys():
                for rect_id_to_remove in data_removals_dict_rectum[UID]:
                    for rect in filtered_rectums:
                        if rect.ROIName == rect_id_to_remove:
                            filtered_rectums.remove(rect)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Rect: {rect_id_to_remove})) ", live_display)

            rectum_ref = [{"ROI":x.ROIName,
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[3],
                        "Raw contour pts zslice list": None,
                        "Raw contour pts": None,
                        "Equal num zslice contour pts": None,
                        "Intra-slice interpolation information": None,
                        "Inter-slice interpolation information": None,
                        "Point cloud raw": None,
                        "Delaunay triangulation global structure": None,
                        "Delaunay triangulation zslice-wise list": None,
                        "Structure centroid pts": None,
                        "Best fit line of centroid pts": None,
                        "Centroid line sample pts": None,
                        "Structure global centroid": None,
                        "Reconstructed structure pts arr": None,
                        "Interpolated structure point cloud dict": None,
                        "Reconstructed structure delaunay global": None,
                        "Maximum pairwise distance": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Structure volume": None,
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None,
                        "MC data: Generated normal dist random samples arr": None,
                        "KDtree": None,
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_rectums)]

            filtered_urethras = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in urethra_list)]

            ### Remove unwanted data (urethra)
            if UID in data_removals_dict_urethra.keys():
                for uret_id_to_remove in data_removals_dict_urethra[UID]:
                    for uret in filtered_urethras:
                        if uret.ROIName == uret_id_to_remove:
                            filtered_urethras.remove(uret)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Uret: {uret_id_to_remove})) ", live_display)

            urethra_ref = [{"ROI":x.ROIName,
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[4],
                        "Raw contour pts zslice list": None,
                        "Raw contour pts": None,
                        "Equal num zslice contour pts": None,
                        "Intra-slice interpolation information": None,
                        "Inter-slice interpolation information": None,
                        "Point cloud raw": None,
                        "Delaunay triangulation global structure": None,
                        "Delaunay triangulation zslice-wise list": None,
                        "Structure centroid pts": None,
                        "Best fit line of centroid pts": None,
                        "Centroid line sample pts": None,
                        "Structure global centroid": None,
                        "Reconstructed structure pts arr": None,
                        "Interpolated structure point cloud dict": None,
                        "Reconstructed structure delaunay global": None,
                        "Maximum pairwise distance": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Structure volume": None,
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None,
                        "MC data: Generated normal dist random samples arr": None,
                        "KDtree": None,
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_urethras)]

            filtered_BXs = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in Bx_list)]

            ### Remove unwanted data (bx)
            if UID in data_removals_dict_bx.keys():
                for bx_id_to_remove in data_removals_dict_bx[UID]:
                    for bpsy in filtered_BXs:
                        if bpsy.ROIName == bx_id_to_remove:
                            filtered_BXs.remove(bpsy)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Bx: {bx_id_to_remove})) ", live_display)


            bpsy_ref = [{"ROI": x.ROIName,
                         "Ref #": x.ROINumber,
                         "Index number": idx,
                         "Struct type": st_ref_list[0],
                         "Simulated bool": False,
                         "Simulated type": 'Real',
                         "Reconstructed biopsy cylinder length (from contour data)": None,
                         "Raw contour pts zslice list": None,
                         "Raw contour pts": None,
                         "Centroid variation arr": None,
                         "Mean centroid variation": None,
                         "Maximum projected distance between original centroids": None,
                         "Equal num zslice contour pts": None,
                         "Intra-slice interpolation information": None,
                         "Inter-slice interpolation information": None,
                         "Point cloud raw": None,
                         "Delaunay triangulation global structure": None,
                         "Delaunay triangulation zslice-wise list": None,
                         "Structure global centroid": None,
                         "Structure centroid pts": None,
                         "Best fit line of centroid pts": None,
                         "Centroid line sample pts": None,
                         "Centroid line unit vec (bx needle base to bx needle tip)": None,
                         "Interpolated structure point cloud dict": None,
                         "Reconstructed structure pts arr": None,
                         "Reconstructed structure point cloud": None,
                         "Reconstructed structure delaunay global": None,
                         "Maximum pairwise distance": None,
                         "Structure volume": None,
                         "Voxel size for structure volume calc": None,
                         "Target DIL dict": None,
                         "Random uniformly sampled volume pts arr": None,
                         "Random uniformly sampled volume pts pcd": None,
                         "Random uniformly sampled volume pts bx coord sys arr": None,
                         "Random uniformly sampled volume pts bx coord sys pcd": None,
                         "Bounding box for random uniformly sampled volume pts": None,
                         "Num sampled bx pts": None,
                         "Uncertainty data": None,
                         "MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr": None,
                         "MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr": None,
                         "MC data: Generated normal dist random samples arr": None,
                         "MC data: Total rigid shift vectors arr": None,
                         "MC data: bx only shifted 3darr": None,
                         "MC data: bx and structure shifted dict": None,
                         "MC data: MC sim translation results dict": None,
                         "MC data: MC sim containment raw results dataframe": None,
                         "MC data: MC sim compiled distances global dataframe": None,
                         "MC data: MC sim compiled distances point-wise dataframe": None,
                         "MC data: MC sim compiled distances voxel-wise dataframe": None,
                         "MC data: MC sim containment and distance all trials dataframe (light)": None,
                         "MC data: compiled sim results dataframe": None,
                         "MC data: compiled sim sum-to-one results dataframe": None,
                         #"MC data: mutual compiled sim results dataframe": None,
                         "MC data: compiled sim results": None,
                         "MC data: mutual compiled sim results": None,
                         #"MC data: tumor tissue probability": None,
                         #"MC data: miss structure tissue probability": None,
                         #"MC data: tissue length above threshold dict": None,
                         "MC data: voxelized containment results dict": None,
                         "MC data: voxelized containment results dict (dict of lists)": None,
                         "MC data: bx to dose NN search objects list": None,
                         #"MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)": None,
                         "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)": None,
                         #"MC data: Dose vals for each sampled bx pt arr (all MC trials)": None,
                         #"MC data: Dose vals for each sampled bx pt arr (nominal)": None,
                         "MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)": None,
                         "MC data: Differential DVH dict": None,
                         "MC data: Cumulative DVH dict": None,
                         "MC data: dose volume metrics dict": None,
                         "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None,
                         "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None,
                         "MC data: voxelized dose results list": None,
                         "MC data: voxelized dose results dict (dict of lists)": None,
                         "Output csv file paths dict": {},
                         "Output data frames": {"Dose output Z and radius": None,
                                                "Dose output voxelized": None,
                                                "Point-wise dose output by MC trial number": None,
                                                "Voxel-wise dose output by MC trial number": None,
                                                #"Mutual containment output by bx point": None,
                                                "Differential DVH by MC trial": None,
                                                "Cumulative DVH by MC trial": None},
                         "Output dicts for data frames": {},
                         "KDtree": None,
                         "Nearest neighbours objects": []
                         } for idx, x in enumerate(filtered_BXs)]





            bpsy_ref_index_start = len(bpsy_ref)
            patient_fraction_number = misc_tools.extract_number_from_string(str(structure_item[0x0010,0x0020].value), fraction_prefixes)
            if simulated_biopsy_fraction_numbers_to_create == 'all':
                create_simulated_biopsies_for_this_fraction = True
            elif type(simulated_biopsy_fraction_numbers_to_create) in [list, tuple, set]:
                create_simulated_biopsies_for_this_fraction = patient_fraction_number in simulated_biopsy_fraction_numbers_to_create
            else:
                create_simulated_biopsies_for_this_fraction = patient_fraction_number == simulated_biopsy_fraction_numbers_to_create
            sim_bpsy_ref_index_start = 0
            bpsy_ref_simulated_total = []
            for bx_sim_type_str, bx_sim_type_dict in bx_sim_locations_dict.items():
                if bx_sim_type_dict["Create"] == True and create_simulated_biopsies_for_this_fraction == True:
                    sim_bx_relative_to = bx_sim_type_dict["Relative to struct type"]
                    bx_sim_ref_identifier_str = bx_sim_type_dict["Identifier string"]

                    sim_bx_relative_to_contour_names = structs_referenced_dict[sim_bx_relative_to]["Contour names"]


                    # Determine the appropriate removal list based on the relative structure type.
                    if bx_sim_type_dict["Relative to struct type"] == st_ref_list[2]:
                        removal_list = data_removals_dict_dil.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[0]:
                        removal_list = data_removals_dict_bx.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[1]:
                        removal_list = data_removals_dict_prostate.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[3]:
                        removal_list = data_removals_dict_rectum.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[4]:
                        removal_list = data_removals_dict_urethra.get(UID, [])
                    else:
                        removal_list = []

                    # Now filter simulated biopsy candidates by including the removal condition.
                    filtered_sim_BXs = [
                        x for x in structure_item.StructureSetROISequence
                        if any(contour.lower() in x.ROIName.lower() for contour in sim_bx_relative_to_contour_names)
                        and x.ROIName not in removal_list
                    ]

                    # old doesnt account for data point removals of relative structures for simulating biopsies!
                    #filtered_sim_BXs = [x for x in structure_item.StructureSetROISequence if sim_bx_relative_to.lower() in x.ROIName.lower()]


                    bpsy_ref_simulated = [{"ROI": "Bx_Tr_"+bx_sim_ref_identifier_str+" " + x.ROIName,
                                "Ref #": bx_sim_ref_identifier_str +" "+ x.ROIName,
                                "Index number": bpsy_ref_index_start + sim_bpsy_ref_index_start + idx,
                                "Struct type": st_ref_list[0],
                                "Simulated bool": True,
                                "Simulated type": bx_sim_type_str,
                                "Transport family": bx_sim_type_dict.get("Transport family", "identity"),
                                "Relative structure type": bx_sim_type_dict["Relative to struct type"],
                                "Relative structure name": x.ROIName,
                                "Relative structure ref #": x.ROINumber,
                                "Reconstructed biopsy cylinder length (from contour data)": None,
                                "Raw contour pts zslice list": None,
                                "Raw contour pts": None,
                                "Centroid variation arr": None,
                                "Mean centroid variation": None,
                                "Maximum projected distance between original centroids": None,
                                "Equal num zslice contour pts": None,
                                "Intra-slice interpolation information": None,
                                "Inter-slice interpolation information": None,
                                "Point cloud raw": None,
                                "Delaunay triangulation global structure": None,
                                "Delaunay triangulation zslice-wise list": None,
                                "Structure global centroid": None,
                                "Structure centroid pts": None,
                                "Best fit line of centroid pts": None,
                                "Centroid line sample pts": None,
                                "Centroid line unit vec (bx needle base to bx needle tip)": None,
                                "Interpolated structure point cloud dict": None,
                                "Reconstructed structure pts arr": None,
                                "Reconstructed structure point cloud": None,
                                "Reconstructed structure delaunay global": None,
                                "Maximum pairwise distance": None,
                                "Structure volume": None,
                                "Voxel size for structure volume calc": None,
                                "Target DIL dict": None,
                                "Random uniformly sampled volume pts arr": None,
                                "Random uniformly sampled volume pts pcd": None,
                                "Random uniformly sampled volume pts bx coord sys arr": None,
                                "Random uniformly sampled volume pts bx coord sys pcd": None,
                                "Bounding box for random uniformly sampled volume pts": None,
                                "Num sampled bx pts": None,
                                "Uncertainty data": None,
                                "MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr": None,
                                "MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr": None,
                                "MC data: Generated normal dist random samples arr": None,
                                "MC data: Total rigid shift vectors arr": None,
                                "MC data: bx only shifted 3darr": None,
                                "MC data: bx and structure shifted dict": None,
                                "MC data: MC sim translation results dict": None,
                                "MC data: MC sim containment raw results dataframe": None,
                                "MC data: MC sim compiled distances global dataframe": None,
                                "MC data: MC sim compiled distances point-wise dataframe": None,
                                "MC data: MC sim compiled distances voxel-wise dataframe": None,
                                "MC data: MC sim containment and distance all trials dataframe (light)": None,
                                "MC data: compiled sim results dataframe": None,
                                "MC data: compiled sim sum-to-one results dataframe": None,
                                #"MC data: mutual compiled sim results dataframe": None,
                                "MC data: compiled sim results": None,
                                "MC data: mutual compiled sim results": None,
                                #"MC data: tumor tissue probability": None,
                                #"MC data: miss structure tissue probability": None,
                                #"MC data: tissue length above threshold dict": None,
                                "MC data: voxelized containment results dict": None,
                                "MC data: voxelized containment results dict (dict of lists)": None,
                                "MC data: bx to dose NN search objects list": None,
                                #"MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)": None,
                                "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)": None,
                                #"MC data: Dose vals for each sampled bx pt arr (all MC trials)": None,
                                #"MC data: Dose vals for each sampled bx pt arr (nominal)": None,
                                "MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)": None,
                                "MC data: Differential DVH dict": None,
                                "MC data: Cumulative DVH dict": None,
                                "MC data: dose volume metrics dict": None,
                                "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None,
                                "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None,
                                "MC data: voxelized dose results list": None,
                                "MC data: voxelized dose results dict (dict of lists)": None,
                                "Simulated biopsy transport request dict": None,
                                "Output csv file paths dict": {},
                                "Output data frames": {"Dose output Z and radius": None,
                                                       "Dose output voxelized": None,
                                                       "Point-wise dose output by MC trial number": None,
                                                       "Voxel-wise dose output by MC trial number": None,
                                                       #"Mutual containment output by bx point": None,
                                                       "Differential DVH by MC trial": None},
                                "Output dicts for data frames": {},
                                "KDtree": None,
                                "Nearest neighbours objects": []
                                } for idx, x in enumerate(filtered_sim_BXs)]

                    sim_bpsy_ref_index_start = len(bpsy_ref_simulated)

                    bpsy_ref_simulated_total = bpsy_ref_simulated_total + bpsy_ref_simulated
                else:
                    pass



            bpsy_ref = bpsy_ref + bpsy_ref_simulated_total

            """
            ## for each reference type, store each item's index number
            for index, item in enumerate(bpsy_ref):
                item["Index number"] = index
            for index, item in enumerate(DIL_ref):
                item["Index number"] = index
            for index, item in enumerate(OAR_ref):
                item["Index number"] = index
            """

            # Note that all of the dataframes in the below "Multi-structure output data frames dict" are output as csvs in the final report
            all_ref = {"Multi-structure information dict (not for csv output)": {"Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe": None, # THIS DATAFRAME WAS DELETED AFTER OPTIMIZATION TO SAVE MEMORY!
                                                                                  },
                        "Multi-structure pre-processing output dataframes dict": {"Selected structures": None,
                                                                                  "Biopsy basic spatial features dataframe": None,
                                                                                  "Simulated biopsy preparation dataframe": None,
                                                                                  #"Structure information dimension": pandas.DataFrame(),
                                                                                  #"Structure information (Non-BX)": pandas.DataFrame(),
                                                                                  "Nearest DILs info dataframe": None,
                                                                                  "Biopsy optimization - Cumulative projection (all points within prostate) dataframe": None,
                                                                                  "Biopsy optimization - DIL centroids optimal targeting dataframe": None,
                                                                                  "Biopsy optimization - Optimal DIL targeting dataframe": None,
                                                                                  "Biopsy optimization - Optimal DIL targeting entire lattice dataframe": None,
                                                                                  TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY: None,
                                                                                  TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY: None,
                                                                                  "Biopsy optimization - Guidance-map firing depth recommendations dataframe": None,
                                                                                  "3D radiomic features all OAR and DIL structures": None,
                                                                                  "Per sample point prostate double sextant classification": None,
                                                                                  "Per voxel prostate double sextant classification": None,
                                                                                  "Simulated biopsy planned vs realized centroid variation validation": None,
                                                                                  "Prostate only points MR ADC dataframe (temporary for pre-processing)": None,
                                                                                  "MR - ADC - summary statistics by structure dataframe": None},
                        "Multi-structure MC simulation output dataframes dict": {"All MC structure transformation values": None,
                                                                                 "Tissue class - Global tissue class statistics": None,
                                                                                 "Tissue class - Global tissue by structure statistics": None,
                                                                                 "Tissue class - Tissue length above threshold": None,
                                                                                 "Tissue class - sum-to-one mc results": None,
                                                                                 "Tissue class - distances global results": None,
                                                                                 "Tissue class - distances pt-wise results": None,
                                                                                 "Tissue class - distances voxel-wise results": None,
                                                                                 "Tissue class - containment and distances (light) results": None,
                                                                                 #"Tissue class - Pt wise mutual tissue class results": None,
                                                                                 "Tissue class - Pt wise structure specific results": None,
                                                                                 #"Dosimetry - All points and trials": pandas.DataFrame(),
                                                                                 "DVH metrics (Dx, Vx) statistics": None,
                                                                                 "MR - " + str(mr_global_multi_structure_output_dataframe_str): None,
                                                                                 "MR - " + str(mr_global_by_voxel_multi_structure_output_dataframe_str): None},
                        }



            # Dictionary comprehension to count occurrences of each type of biopsy
            bpsy_type_counts_dict = {item["Simulated type"]: sum(1 for d in bpsy_ref if d["Simulated type"] == item["Simulated type"]) for item in bpsy_ref}

            bpsy_info = {"Num structs": len(bpsy_ref),
                         "Num sim structs": len(bpsy_ref_simulated_total),
                         "Num real structs": len(bpsy_ref) - len(bpsy_ref_simulated_total),
                         "Biopsy type counts": bpsy_type_counts_dict}
            OAR_info = {"Num structs": len(OAR_ref)}
            DIL_info = {"Num structs": len(DIL_ref)}
            rectum_info = {"Num structs": len(rectum_ref)}
            urethra_info = {"Num structs": len(urethra_ref)}
            patient_total_num_structs = bpsy_info["Num structs"] + OAR_info["Num structs"] + DIL_info["Num structs"] + rectum_info["Num structs"] + urethra_info["Num structs"]
            all_structs_info = {"Total num structs": patient_total_num_structs}

            global_num_OAR = global_num_OAR + OAR_info["Num structs"]
            global_num_DIL = global_num_DIL + DIL_info["Num structs"]
            global_num_biopsies = global_num_biopsies + bpsy_info["Num structs"]
            global_total_num_structs = global_total_num_structs + patient_total_num_structs
            global_num_cases = global_num_cases + 1
            if str(structure_item[0x0010,0x0010].value) not in global_unique_patient_names_list:
                global_unique_patient_names_list.append(str(structure_item[0x0010,0x0010].value))

            master_st_ds_ref_dict[UID] = {"Patient UID (generated)": str(UID),
                                            "Patient ID (from dicom)": str(structure_item[0x0010,0x0020].value),
                                            "Patient Name": str(structure_item[0x0010,0x0010].value),
                                            "Fraction number": misc_tools.extract_number_from_string(str(structure_item[0x0010,0x0020].value), fraction_prefixes),
                                            st_ref_list[0]: bpsy_ref,
                                            st_ref_list[1]: OAR_ref,
                                            st_ref_list[2]: DIL_ref,
                                            st_ref_list[3]: rectum_ref,
                                            st_ref_list[4]: urethra_ref,
                                            all_ref_key: all_ref,
                                            "Ready to plot data list": None
                                        }

            master_st_ds_info_dict[UID] = {"Patient UID (generated)": str(UID),
                                            "Patient ID (from dicom)": str(structure_item[0x0010,0x0020].value),
                                            "Patient Name": str(structure_item[0x0010,0x0010].value),
                                            "Fraction number": misc_tools.extract_number_from_string(str(structure_item[0x0010,0x0020].value), fraction_prefixes),
                                            st_ref_list[0]: bpsy_info,
                                            st_ref_list[1]: OAR_info,
                                            st_ref_list[2]: DIL_info,
                                            st_ref_list[3]: rectum_info,
                                            st_ref_list[4]: urethra_info,
                                            all_ref_key: all_structs_info
                                        }

    ### Dosimetry
    for UID, dose_item_path in dose_dcm_dict.items():
        if master_st_ds_ref_dict[UID]["Fraction number"] == 1:
            continue
        with pydicom.dcmread(dose_item_path, defer_size = '2 MB') as dose_item:
            dose_ID = UID + dose_item.StudyDate
            dose_ref_dict = {"Dose ID": dose_ID,
                             "Study date": dose_item.StudyDate,
                             "Dose pixel data": dose_item.PixelData,
                             "Dose pixel arr": dose_item.pixel_array,
                             "Pixel spacing": [float(item) for item in dose_item.PixelSpacing],
                             "Dose grid scaling": float(dose_item.DoseGridScaling),
                             "Dose units": dose_item.DoseUnits,
                             "Dose type": dose_item.DoseType,
                             "Grid frame offset vector": [float(item) for item in dose_item.GridFrameOffsetVector],
                             "Image orientation patient": [float(item) for item in dose_item.ImageOrientationPatient],
                             "Image position patient": [float(item) for item in dose_item.ImagePositionPatient],
                             #"Dose phys space and pixel 3d arr": None,
                             "Dose and gradient phys space and pixel 3d arr": None,
                             "Dose grid point cloud": None,
                             "Dose grid point cloud thresholded": None,
                             "Dose grid gradient point cloud": None,
                             "Dose grid gradient point cloud thresholded": None,
                             "KDtree": None,
                             "KDtree gradient": None
                             }
            master_st_ds_ref_dict[UID][ds_ref] = dose_ref_dict

    """
    ### MR ADC
    for UID, mr_adc_item_paths_list in MR_ADC_dcms_dict.items():
        mr_adc_ref_dict = {}
        for mr_adc_item_path in mr_adc_item_paths_list:
            with pydicom.dcmread(mr_adc_item_path, defer_size = '2 MB') as mr_adc_item:
                seriesinstanceUID = mr_adc_item.SeriesInstanceUID
                mr_adc_ID = UID + mr_adc_item.StudyDate



                if len(mr_adc_item.RealWorldValueMappingSequence) > 1:
                    important_info.add_text_line(f"Multiple real world value mappings detected for ({UID}, {mr_adc_ID})) ", live_display)
                if seriesinstanceUID not in mr_adc_ref_dict:
                    mr_adc_ref_subdict = {"MR ADC ID": mr_adc_ID,
                                    "Series instance UID": seriesinstanceUID,
                                    "Study date": mr_adc_item.StudyDate,
                                    "Pixel arr (all slices)": mr_adc_item.pixel_array,
                                    "Pixel spacing": np.array(mr_adc_item.PixelSpacing),
                                    "Units": str(mr_adc_item.RealWorldValueMappingSequence[0].MeasurementUnitsCodeSequence[0].CodeMeaning),
                                    "RWVSlope (all slices)": np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueSlope),
                                    "RWVIntercept (all slices)": np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueIntercept),
                                    "RWV Units": mr_adc_item.RealWorldValueMappingSequence[0].LUTLabel,
                                    "Slice thickness":  mr_adc_item.SliceThickness,
                                    "Image orientation patient": np.array(mr_adc_item.ImageOrientationPatient),
                                    "Image position patient (all slices)": np.array(mr_adc_item.ImagePositionPatient),
                                    "MR ADC phys space Nx4 arr": None,
                                    "MR ADC phys space Nx4 arr (filtered, non-negative)": None,
                                    "MR ADC grid point cloud": None,
                                    "MR ADC grid point cloud thresholded": None,
                                    "KDtree": None
                                    }
                    mr_adc_ref_dict[seriesinstanceUID] = mr_adc_ref_subdict
                else:
                    mr_adc_ref_subdict = mr_adc_ref_dict[seriesinstanceUID]
                    mr_adc_ref_subdict["Pixel arr (all slices)"] = np.dstack((mr_adc_ref_subdict["Pixel arr (all slices)"], mr_adc_item.pixel_array))
                    mr_adc_ref_subdict["RWVSlope (all slices)"] = np.hstack((mr_adc_ref_subdict["RWVSlope (all slices)"], np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueSlope)))
                    mr_adc_ref_subdict["RWVIntercept (all slices)"] = np.hstack((mr_adc_ref_subdict["RWVIntercept (all slices)"], np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueIntercept)))
                    mr_adc_ref_subdict["Image position patient (all slices)"] = np.vstack((mr_adc_ref_subdict["Image position patient (all slices)"], np.array(mr_adc_item.ImagePositionPatient)))
                    mr_adc_ref_dict[seriesinstanceUID] = mr_adc_ref_subdict

        master_st_ds_ref_dict[UID][mr_adc_ref] = mr_adc_ref_dict

    """

    ### MR ADC, updated to account for missing RealWorldValueMappingSequence
    for UID, mr_adc_item_paths_list in MR_ADC_dcms_dict.items():
        mr_adc_ref_dict = {}
        for mr_adc_item_path in mr_adc_item_paths_list:
            with pydicom.dcmread(mr_adc_item_path, defer_size='2 MB') as mr_adc_item:
                seriesinstanceUID = mr_adc_item.SeriesInstanceUID
                mr_adc_ID = UID + mr_adc_item.StudyDate

                rwvm = getattr(mr_adc_item, "RealWorldValueMappingSequence", None)

                # Handle multiple RWVMs
                if rwvm and len(rwvm) > 1:
                    important_info.add_text_line(
                        f"Multiple real world value mappings detected for ({UID}, {mr_adc_ID})", live_display
                    )

                # Safely extract or assign defaults
                if rwvm and len(rwvm) > 0:
                    rwv = rwvm[0]
                    units = str(getattr(rwv.MeasurementUnitsCodeSequence[0], "CodeMeaning", "unknown"))
                    slope = np.array(rwv.RealWorldValueSlope)
                    intercept = np.array(rwv.RealWorldValueIntercept)
                    rwv_units = getattr(rwv, "LUTLabel", "unknown")
                else:
                    units = "mm\u00B2/s (assumed)"
                    slope = np.array([1e-6])  # Default slope for mm2/s
                    intercept = np.array([0.0])
                    rwv_units = "mm\u00B2/s (assumed)"
                    important_info.add_text_line(
                        f"No RealWorldValueMappingSequence found for ({UID}, {mr_adc_ID}) – using defaults.",
                        live_display
                    )

                if seriesinstanceUID not in mr_adc_ref_dict:
                    mr_adc_ref_subdict = {
                        "MR ADC ID": mr_adc_ID,
                        "Series instance UID": seriesinstanceUID,
                        "Study date": mr_adc_item.StudyDate,
                        "Pixel arr (all slices)": mr_adc_item.pixel_array,
                        "Pixel spacing": np.array(mr_adc_item.PixelSpacing),
                        "Units": units,
                        "RWVSlope (all slices)": slope,
                        "RWVIntercept (all slices)": intercept,
                        "RWV Units": rwv_units,
                        "Slice thickness": getattr(mr_adc_item, "SliceThickness", -1),
                        "Image orientation patient": np.array(mr_adc_item.ImageOrientationPatient),
                        "Image position patient (all slices)": np.array(mr_adc_item.ImagePositionPatient),
                        "MR ADC phys space Nx4 arr": None,
                        "MR ADC phys space Nx4 arr (filtered, non-negative)": None,
                        "MR ADC grid point cloud": None,
                        "MR ADC grid point cloud thresholded": None,
                        "KDtree": None
                    }
                    mr_adc_ref_dict[seriesinstanceUID] = mr_adc_ref_subdict
                else:
                    mr_adc_ref_subdict = mr_adc_ref_dict[seriesinstanceUID]
                    mr_adc_ref_subdict["Pixel arr (all slices)"] = np.dstack(
                        (mr_adc_ref_subdict["Pixel arr (all slices)"], mr_adc_item.pixel_array)
                    )
                    mr_adc_ref_subdict["RWVSlope (all slices)"] = np.hstack(
                        (mr_adc_ref_subdict["RWVSlope (all slices)"], slope)
                    )
                    mr_adc_ref_subdict["RWVIntercept (all slices)"] = np.hstack(
                        (mr_adc_ref_subdict["RWVIntercept (all slices)"], intercept)
                    )
                    mr_adc_ref_subdict["Image position patient (all slices)"] = np.vstack(
                        (mr_adc_ref_subdict["Image position patient (all slices)"], np.array(mr_adc_item.ImagePositionPatient))
                    )
                    mr_adc_ref_dict[seriesinstanceUID] = mr_adc_ref_subdict

                pass
        pass

        master_st_ds_ref_dict[UID][mr_adc_ref] = mr_adc_ref_dict





    ### Plan file
    for UID, plan_item_path in plan_dcm_dict.items():
        with pydicom.dcmread(plan_item_path, defer_size = '2 MB') as plan_item:
            plan_ID = UID + plan_item.StudyDate
            plan_ref_dict = {"Plan ID": plan_ID,
                             "Study date": plan_item.StudyDate,
                             "Dose units": 'Gy', # this is by default for this dicom tag: (300A,0026)
                             "Prescription doses dict": {}
                             }

            for dose_ref_seq_ind in range(len(plan_item.DoseReferenceSequence)):
                plan_ref_dict["Prescription doses dict"][plan_item.DoseReferenceSequence[dose_ref_seq_ind]["DoseReferenceType"].value] = plan_item.DoseReferenceSequence[dose_ref_seq_ind]["TargetPrescriptionDose"].value

            master_st_ds_ref_dict[UID][pln_ref] = plan_ref_dict

    preprocessing_info = {"Interslice interp dist": interp_inter_slice_dist,
                          "Intraslice interp dist": interp_intra_slice_dist,
                          "Preprocessing performed": False,}

    mc_info = {"Num MC containment simulations": None,
               "Num MC dose simulations": None,
               "Num MC MR simulations": None,
               "Num optimizer v2 transform samples": None,
               "Num stochastic targeting transform samples": None,
               "Num sample pts per BX core": None,
               "BX sample pt lattice spacing (mm)": None,
               "BX sample pt volume element (mm^3)": None,
               "Max of num MC simulations": None,
               "Max of generated transform samples": None,
               'MC sim performed': False,
               'MC containment sim performed': False,
               'MC dose sim performed': False,
               'MC MR sim performed': False,
               }

    random_info = {"Transform generation random seed": None,
                   "Optimizer v1 random seed": None,
                   }

    # count number of biopsies of each type for the entire cohort
    list_of_bpsy_nums_by_bpsy_type_all_patients = [pt_sp_dict[st_ref_list[0]]["Biopsy type counts"] for pt_sp_dict in master_st_ds_info_dict.values()]
    global_num_biopsies_by_type = {key: sum(d[key] for d in list_of_bpsy_nums_by_bpsy_type_all_patients if key in d) for d in list_of_bpsy_nums_by_bpsy_type_all_patients for key in d}

    # Determine the types of biopsies made
    bx_types_list = ['Real'] + [key for key, value in bx_sim_locations_dict.items() if value.get("Create", False)]

    master_st_ds_info_global_dict["Global"] = {"Num cases": global_num_cases, # This is a mixture of patients and fractions, ie. 181 F1 and 181 F2 are two different cases
                                               "Num unique patient names": len(global_unique_patient_names_list),
                                               "Num structures": global_total_num_structs,
                                               "Num biopsies": global_num_biopsies,
                                               "Num biopsies by bx type dict": global_num_biopsies_by_type,
                                               "Num DILs": global_num_DIL,
                                               "Bx types list": bx_types_list,
                                               "Preprocessing info": preprocessing_info,
                                               "MC info": mc_info,
                                               "Random info": random_info,
                                               'Patient specific guidance map figures directory dict': None,
                                               'Guidance map figures dir': None,
                                               "Specific output dir": None
                                               }

    master_st_ds_info_global_dict["By patient"] = master_st_ds_info_dict
    return master_st_ds_ref_dict, master_st_ds_info_global_dict
