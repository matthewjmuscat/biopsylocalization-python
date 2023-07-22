import pandas
import plotly.express as px
import plotting_funcs
import numpy as np
import math_funcs as mf
import plotly.graph_objects as go
import misc_tools

def production_plot_sampled_shift_vector_box_plots_by_patient(patientUID,
                                              patient_sp_output_figures_dir_dict,
                                              structs_referenced_list,
                                              bx_structs,
                                              pydicom_item,
                                              svg_image_scale,
                                              svg_image_width,
                                              svg_image_height):
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    structure_name_and_shift_type_dict_for_pandas_data_frame = {}
    for structs in structs_referenced_list:
        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
            structureID = specific_structure["ROI"]
            structure_reference_number = specific_structure["Ref #"]
            if structs == bx_structs:
                sampled_rigid_shifts_from_normal_and_uniform_distribution = specific_structure["MC data: Total rigid shift vectors arr"]
                sampled_rigid_shifts_from_normal_and_uniform_distribution_magnitude = np.linalg.norm(sampled_rigid_shifts_from_normal_and_uniform_distribution, axis = 1)
                sample_description = 'Total translation (length uncertainty + normal)'
                structure_name_and_shift_type_dict_for_pandas_data_frame[str(structureID) + ' '+ sample_description] = sampled_rigid_shifts_from_normal_and_uniform_distribution_magnitude
            # create box plots of sampled rigid shifts for each structure                      
            sampled_rigid_shifts_from_normal_distribution = specific_structure['MC data: Generated normal dist random samples arr']
            sampled_rigid_shifts_from_normal_distribution_magnitude = np.linalg.norm(sampled_rigid_shifts_from_normal_distribution, axis = 1)
            sample_description = 'Rigid translation (normal)'
            structure_name_and_shift_type_dict_for_pandas_data_frame[str(structureID) + ' '+ sample_description] = sampled_rigid_shifts_from_normal_distribution_magnitude
    
    structure_name_and_shift_type_dict_pandas_data_frame = pandas.DataFrame(data=structure_name_and_shift_type_dict_for_pandas_data_frame)

    
    
    # box plot
    fig = px.box(structure_name_and_shift_type_dict_pandas_data_frame, points = False)
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Sampled shift magnitude (mm)',
        xaxis_title='Structure',
        title='Sampled translation magnitudes (' + patientUID +')',
        hovermode="x unified"
    )

    svg_dose_fig_name = patientUID + ' - sampled_translations_magnitudes_box_plot.svg'
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = patientUID + ' - sampled_translations_magnitudes_box_plot.html'
    html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path)



def production_plot_axial_dose_distribution_all_trials_and_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                                   patientUID,
                                                                   pydicom_item,
                                                                   bx_structs,
                                                                   global_regression_ans,
                                                                   regression_type_ans,
                                                                   parallel_pool,
                                                                   num_bootstraps_all_MC_data_input,
                                                                   NPKR_bandwidth,
                                                                   svg_image_scale,
                                                                   svg_image_width,
                                                                   svg_image_height,
                                                                   num_z_vals_to_evaluate_for_regression_plots
                                                                   ):
    # generate pandas data frame by reading dose output from file
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
        mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"].copy()
        std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"].copy()
        quantiles_dose_val_specific_bx_pt_dict_of_lists = stats_dose_val_all_MC_trials_by_bx_pt_list["Qunatiles dose by bx pt dict"].copy()
        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)


        # create a 2d scatter plot with all MC trials on plot
        dose_vals_all_MC_trials_by_sampled_bx_pt_list = specific_bx_structure['MC data: Dose vals for each sampled bx pt list']
        pt_radius_point_wise_for_pd_data_frame_list = []
        axial_Z_point_wise_for_pd_data_frame_list = []
        dose_vals_point_wise_for_pd_data_frame_list = []
        MC_trial_index_point_wise_for_pd_data_frame_list = []
        for pt_index, specific_pt_all_MC_dose_vals in enumerate(dose_vals_all_MC_trials_by_sampled_bx_pt_list):
            pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + [pt_radius_bx_coord_sys]*len(specific_pt_all_MC_dose_vals)
            axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + [bx_points_bx_coords_sys_arr[pt_index,2]]*len(specific_pt_all_MC_dose_vals)
            dose_vals_point_wise_for_pd_data_frame_list = dose_vals_point_wise_for_pd_data_frame_list + specific_pt_all_MC_dose_vals
            MC_trial_index_point_wise_for_pd_data_frame_list = MC_trial_index_point_wise_for_pd_data_frame_list + list(range(1,len(specific_pt_all_MC_dose_vals) + 1))
        
        dose_output_dict_by_MC_trial_for_pandas_data_frame = {"Radial pos (mm)": pt_radius_point_wise_for_pd_data_frame_list, "Axial pos Z (mm)": axial_Z_point_wise_for_pd_data_frame_list, "Dose (Gy)": dose_vals_point_wise_for_pd_data_frame_list, "MC trial num": MC_trial_index_point_wise_for_pd_data_frame_list}
        
        dose_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=dose_output_dict_by_MC_trial_for_pandas_data_frame)

        # do non parametric kernel regression (local linear)
        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=num_z_vals_to_evaluate_for_regression_plots)
        
        if regression_type_ans == True and global_regression_ans == True:
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                parallel_pool,
                x = dose_output_dict_by_MC_trial_for_pandas_data_frame["Axial pos Z (mm)"], 
                y = dose_output_dict_by_MC_trial_for_pandas_data_frame["Dose (Gy)"], 
                eval_x = z_vals_to_evaluate, N=num_bootstraps_all_MC_data_input, conf_interval=0.95
            )
        elif regression_type_ans == False and global_regression_ans == True:
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                parallel_pool,
                x = dose_output_dict_by_MC_trial_for_pandas_data_frame["Axial pos Z (mm)"], 
                y = dose_output_dict_by_MC_trial_for_pandas_data_frame["Dose (Gy)"], 
                eval_x = z_vals_to_evaluate, N=num_bootstraps_all_MC_data_input, conf_interval=0.95, bandwidth = NPKR_bandwidth
            )
        elif global_regression_ans == False:
            pass
        
        
        # create 2d scatter dose plot axial (z) vs all doses from all MC trials
                                
        fig_global = px.scatter(dose_output_by_MC_trial_pandas_data_frame, x="Axial pos Z (mm)", y="Dose (Gy)", color = "MC trial num", width  = svg_image_width, height = svg_image_height)
        if global_regression_ans == True:
            fig_global.add_trace(
                go.Scatter(
                    name='Regression',
                    x=z_vals_to_evaluate,
                    y=all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit,
                    mode="lines",
                    line=dict(color='rgb(31, 119, 180)'),
                    showlegend=True
                    )
            )
            fig_global.add_trace(
                go.Scatter(
                    name='Upper 95% CI',
                    x=z_vals_to_evaluate,
                    y=all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=True
                )
            )
            fig_global.add_trace(
                go.Scatter(
                    name='Lower 95% CI',
                    x=z_vals_to_evaluate,
                    y=all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(0, 100, 20, 0.3)',
                    fill='tonexty',
                    showlegend=True
                )
            )
        elif global_regression_ans == False:
            pass
        
        fig_global.update_layout(
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )

        fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)

        if global_regression_ans == True:
            fig_regression_only = go.Figure([
                go.Scatter(
                    name='Regression',
                    x=z_vals_to_evaluate,
                    y=all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit,
                    mode="lines",
                    line=dict(color='rgb(31, 119, 180)'),
                    showlegend=True
                    ),
                go.Scatter(
                    name='Upper 95% CI',
                    x=z_vals_to_evaluate,
                    y=all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=True
                ),
                go.Scatter(
                    name='Lower 95% CI',
                    x=z_vals_to_evaluate,
                    y=all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(0, 100, 20, 0.3)',
                    fill='tonexty',
                    showlegend=True
                )
            ])
            fig_regression_only.update_layout(
                yaxis_title='Conditional mean dose (Gy)',
                xaxis_title='Axial pos Z (mm)',
                title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                hovermode="x unified"
            )
            fig_regression_only = plotting_funcs.fix_plotly_grid_lines(fig_regression_only, y_axis = True, x_axis = True)
        elif global_regression_ans == False:
            pass
                                
        svg_all_MC_trials_dose_fig_name = bx_struct_roi + ' - 2d_scatter_all_MC_trials_dose.svg'
        svg_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_dose_fig_name)
        fig_global.write_image(svg_all_MC_trials_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_all_MC_trials_dose_fig_name = bx_struct_roi + ' - 2d_scatter_all_MC_trials_dose.html'
        html_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_dose_fig_name)
        fig_global.write_html(html_all_MC_trials_dose_fig_file_path)
        if global_regression_ans == True:
            svg_all_MC_trials_dose_fig_name = bx_struct_roi + ' - 2d_regression_all_MC_trials_dose.svg'
            svg_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_dose_fig_name)
            fig_regression_only.write_image(svg_all_MC_trials_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_all_MC_trials_dose_fig_name = bx_struct_roi + ' - 2d_regression_all_MC_trials_dose.html'
            html_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_dose_fig_name)
            fig_regression_only.write_html(html_all_MC_trials_dose_fig_file_path)
        elif global_regression_ans == False:
            pass
        
def production_3d_scatter_dose_axial_radial_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                   patientUID,
                                                                   pydicom_item,
                                                                   bx_structs,
                                                                   parallel_pool,
                                                                   svg_image_scale,
                                                                   svg_image_width,
                                                                   svg_image_height
                                                                   ):
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
        mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"].copy()
        std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"].copy()
        quantiles_dose_val_specific_bx_pt_dict_of_lists = stats_dose_val_all_MC_trials_by_bx_pt_list["Qunatiles dose by bx pt dict"].copy()
        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)


        # create a 2d scatter plot with all MC trials on plot
        dose_vals_all_MC_trials_by_sampled_bx_pt_list = specific_bx_structure['MC data: Dose vals for each sampled bx pt list']
        pt_radius_point_wise_for_pd_data_frame_list = []
        axial_Z_point_wise_for_pd_data_frame_list = []
        dose_vals_point_wise_for_pd_data_frame_list = []
        MC_trial_index_point_wise_for_pd_data_frame_list = []
        for pt_index, specific_pt_all_MC_dose_vals in enumerate(dose_vals_all_MC_trials_by_sampled_bx_pt_list):
            pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + [pt_radius_bx_coord_sys]*len(specific_pt_all_MC_dose_vals)
            axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + [bx_points_bx_coords_sys_arr[pt_index,2]]*len(specific_pt_all_MC_dose_vals)
            dose_vals_point_wise_for_pd_data_frame_list = dose_vals_point_wise_for_pd_data_frame_list + specific_pt_all_MC_dose_vals
            MC_trial_index_point_wise_for_pd_data_frame_list = MC_trial_index_point_wise_for_pd_data_frame_list + list(range(1,len(specific_pt_all_MC_dose_vals) + 1))

        # create 3d scatter dose plot axial (z) vs radial (r) vs mean dose
        dose_output_dict_for_pandas_data_frame = {"Radial pos (mm)": pt_radius_bx_coord_sys, "Axial pos Z (mm)": bx_points_bx_coords_sys_arr[:,2], "Mean dose (Gy)": mean_dose_val_specific_bx_pt, "STD dose": std_dose_val_specific_bx_pt}
        dose_output_dict_for_pandas_data_frame.update(quantiles_dose_val_specific_bx_pt_dict_of_lists)
        dose_output_pandas_data_frame = pandas.DataFrame(data=dose_output_dict_for_pandas_data_frame)
        specific_bx_structure["Output data frames"]["Dose output Z and radius"] = dose_output_pandas_data_frame
        
        
            
        fig = px.scatter_3d(dose_output_pandas_data_frame, x="Axial pos Z (mm)", y="Radial pos (mm)", z="Mean dose (Gy)", error_z = "STD dose", width  = svg_image_width, height = svg_image_height)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + ' - 3d_scatter_dose.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + ' - 3d_scatter_dose.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)


        # create 2d scatter dose color map plot axial (z) vs radial (r) vs mean dose (color)
        fig = px.scatter(dose_output_pandas_data_frame, x="Axial pos Z (mm)", y="Radial pos (mm)", color="Mean dose (Gy)", width  = svg_image_width, height = svg_image_height)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + ' - 2d_scatter_axial_radial_color_dose.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + ' - 2d_scatter_axial_radial_color_dose.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)
        
def production_plot_axial_dose_distribution_quantile_regressions_by_patient(patient_sp_output_figures_dir_dict,
                                                                patientUID,
                                                                pydicom_item,
                                                                bx_structs,
                                                                regression_type_ans,
                                                                parallel_pool,
                                                                NPKR_bandwidth,
                                                                svg_image_scale,
                                                                svg_image_width,
                                                                svg_image_height,
                                                                num_z_vals_to_evaluate_for_regression_plots
                                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
        mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"].copy()
        std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"].copy()
        quantiles_dose_val_specific_bx_pt_dict_of_lists = stats_dose_val_all_MC_trials_by_bx_pt_list["Qunatiles dose by bx pt dict"].copy()
        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)


        # create a 2d scatter plot with all MC trials on plot
        dose_vals_all_MC_trials_by_sampled_bx_pt_list = specific_bx_structure['MC data: Dose vals for each sampled bx pt list']
        pt_radius_point_wise_for_pd_data_frame_list = []
        axial_Z_point_wise_for_pd_data_frame_list = []
        dose_vals_point_wise_for_pd_data_frame_list = []
        MC_trial_index_point_wise_for_pd_data_frame_list = []
        for pt_index, specific_pt_all_MC_dose_vals in enumerate(dose_vals_all_MC_trials_by_sampled_bx_pt_list):
            pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + [pt_radius_bx_coord_sys]*len(specific_pt_all_MC_dose_vals)
            axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + [bx_points_bx_coords_sys_arr[pt_index,2]]*len(specific_pt_all_MC_dose_vals)
            dose_vals_point_wise_for_pd_data_frame_list = dose_vals_point_wise_for_pd_data_frame_list + specific_pt_all_MC_dose_vals
            MC_trial_index_point_wise_for_pd_data_frame_list = MC_trial_index_point_wise_for_pd_data_frame_list + list(range(1,len(specific_pt_all_MC_dose_vals) + 1))
        
        dose_output_dict_for_pandas_data_frame = {"Radial pos (mm)": pt_radius_bx_coord_sys, "Axial pos Z (mm)": bx_points_bx_coords_sys_arr[:,2], "Mean dose (Gy)": mean_dose_val_specific_bx_pt, "STD dose": std_dose_val_specific_bx_pt}
        dose_output_dict_for_pandas_data_frame.update(quantiles_dose_val_specific_bx_pt_dict_of_lists)
        dose_output_pandas_data_frame = pandas.DataFrame(data=dose_output_dict_for_pandas_data_frame)
        specific_bx_structure["Output data frames"]["Dose output Z and radius"] = dose_output_pandas_data_frame
        
        
        # do non parametric kernel regression (local linear)
        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=num_z_vals_to_evaluate_for_regression_plots)


        # create 2d scatter dose plot axial (z) vs mean dose 
        #fig = px.scatter(dose_output_pandas_data_frame, x="Axial pos Z (mm)", y="Mean dose (Gy)", error_y = "STD dose", width  = svg_image_width, height = svg_image_height)
        
        # can change the name of this dictionary to 'regression_colors_dict' to make all the regressions a different color
        regression_colors_dict_different = {"Q95":'rgba(255, 0, 0, 1)',
                                    "Q75":'rgba(0, 255, 255, 1)', 
                                    "Q50":'rgba(255, 0, 255, 1)',
                                    'Q25':'rgba(255, 255, 0, 1)',
                                    "Q5":'rgba(0, 0, 255, 1)',
                                    "Mean":'rgba(0, 255, 0, 1)'
        }
        # can change the name of this dictionary to 'regression_colors_dict' to make the paired regressions the same color
        regression_colors_dict = {"Q95":'rgba(255, 0, 0, 1)',
                                    "Q75":'rgba(0, 0, 255, 1)', 
                                    "Q50":'rgba(255, 0, 255, 1)',
                                    'Q25':'rgba(0, 0, 255, 1)',
                                    "Q5":'rgba(255, 0, 0, 1)',
                                    "Mean":'rgba(0, 255, 0, 1)'
        }
        
        regression_line_styles_dict = {"Q95": '4,14',
                                    "Q75": '10,10,10', 
                                    "Q50": 'dot',
                                    'Q25': '12,2,12',
                                    "Q5": '2,16,2',
                                    "Mean": 'solid'
        }
        fig = go.Figure([
            go.Scatter(
                name='Mean',
                x=dose_output_dict_for_pandas_data_frame["Axial pos Z (mm)"],
                y=dose_output_dict_for_pandas_data_frame["Mean dose (Gy)"],
                mode='markers',
                marker_color=regression_colors_dict["Mean"],
                showlegend=True
            ),
            go.Scatter(
                name='95th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Axial pos Z (mm)"],
                y=dose_output_dict_for_pandas_data_frame["Q95"],
                mode='markers',
                marker_color=regression_colors_dict["Q95"],
                showlegend=True
            ),
            go.Scatter(
                name='75th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Axial pos Z (mm)"],
                y=dose_output_dict_for_pandas_data_frame["Q75"],
                mode='markers',
                marker_color=regression_colors_dict["Q75"],
                showlegend=True
            ),
            go.Scatter(
                name='50th Quantile (median)',
                x=dose_output_dict_for_pandas_data_frame["Axial pos Z (mm)"],
                y=dose_output_dict_for_pandas_data_frame["Q50"],
                mode='markers',
                marker_color=regression_colors_dict["Q50"],
                showlegend=True
            ),
            go.Scatter(
                name='25th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Axial pos Z (mm)"],
                y=dose_output_dict_for_pandas_data_frame["Q25"],
                mode='markers',
                marker_color=regression_colors_dict["Q25"],
                showlegend=True
            ),
            go.Scatter(
                name='5th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Axial pos Z (mm)"],
                y=dose_output_dict_for_pandas_data_frame["Q5"],
                mode='markers',
                marker_color=regression_colors_dict["Q5"],
                showlegend=True
            )
        ])
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + ' - 2d_scatter_dose.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + ' - 2d_scatter_dose.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)



        # perform non parametric kernel regression through conditional quantiles and conditional mean doses
        dose_output_dict_for_regression = {"Radial pos (mm)": pt_radius_bx_coord_sys, "Axial pos Z (mm)": bx_points_bx_coords_sys_arr[:,2], "Mean": mean_dose_val_specific_bx_pt, "STD dose": std_dose_val_specific_bx_pt}
        dose_output_dict_for_regression.update(quantiles_dose_val_specific_bx_pt_dict_of_lists)
        non_parametric_kernel_regressions_dict = {}
        data_for_non_parametric_kernel_regressions_dict = {}
        data_keys_to_regress = ["Q95","Q5","Q50","Mean","Q75","Q25"]
        num_bootstraps_mean_and_quantile_data = 15
        for data_key in data_keys_to_regress:
            data_for_non_parametric_kernel_regressions_dict[data_key]=dose_output_dict_for_regression[data_key].copy()
            
        for data_key, data_to_regress in data_for_non_parametric_kernel_regressions_dict.items():
            if regression_type_ans == True:
                non_parametric_regression_fit, \
                non_parametric_regression_lower, \
                non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = dose_output_dict_for_regression["Axial pos Z (mm)"], 
                    y = data_to_regress, 
                    eval_x = z_vals_to_evaluate, 
                    N=num_bootstraps_mean_and_quantile_data, 
                    conf_interval=0.95
                )
            elif regression_type_ans == False:
                non_parametric_regression_fit, \
                non_parametric_regression_lower, \
                non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = dose_output_dict_for_regression["Axial pos Z (mm)"], 
                    y = data_to_regress, 
                    eval_x = z_vals_to_evaluate, 
                    N=num_bootstraps_mean_and_quantile_data, 
                    conf_interval=0.95, bandwidth = NPKR_bandwidth
                )
            
            non_parametric_kernel_regressions_dict[data_key] = (
                non_parametric_regression_fit, 
                non_parametric_regression_lower, 
                non_parametric_regression_upper
            )
            
        # create regression figure
        fig_regressions_only_quantiles_and_mean = go.Figure()
        for data_key,regression_tuple in non_parametric_kernel_regressions_dict.items(): 
            fig_regressions_only_quantiles_and_mean.add_trace(
                go.Scatter(
                    name=data_key+' regression',
                    x=z_vals_to_evaluate,
                    y=regression_tuple[0],
                    mode="lines",
                    line=dict(color = regression_colors_dict[data_key], dash = regression_line_styles_dict[data_key]),
                    showlegend=True
                    )
            )
            fig_regressions_only_quantiles_and_mean.add_trace(
                go.Scatter(
                    name=data_key+': Upper 95% CI',
                    x=z_vals_to_evaluate,
                    y=regression_tuple[2],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                )
            )
            fig_regressions_only_quantiles_and_mean.add_trace(
                go.Scatter(
                    name=data_key+': Lower 95% CI',
                    x=z_vals_to_evaluate,
                    y=regression_tuple[1],
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(0, 100, 20, 0.3)',
                    fill='tonexty',
                    showlegend=False
                )
            )
            
        
        fig_regressions_only_quantiles_and_mean.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )
        fig_regressions_only_quantiles_and_mean = plotting_funcs.fix_plotly_grid_lines(fig_regressions_only_quantiles_and_mean, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + ' - 2d_regressions_quantiles_mean_dose.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig_regressions_only_quantiles_and_mean.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + ' - 2d_regressions_quantiles_mean_dose.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig_regressions_only_quantiles_and_mean.write_html(html_dose_fig_file_path)


        # create simplified regression figure
        quantile_pairs_list = [("Q5","Q95", 'rgba(0, 255, 0, 0.3)'), ("Q25","Q75", 'rgba(0, 0, 255, 0.3)')] # must be organized where first element is lower and second element is upper bound
        fig_regressions_dose_quantiles_simple = go.Figure()
        for quantile_pair_tuple in quantile_pairs_list: 
            lower_regression_key = quantile_pair_tuple[0]
            upper_regression_key = quantile_pair_tuple[1]
            fill_color = quantile_pair_tuple[2]
            lower_regression_tuple = non_parametric_kernel_regressions_dict[lower_regression_key]
            upper_regression_tuple = non_parametric_kernel_regressions_dict[upper_regression_key]
            fig_regressions_dose_quantiles_simple.add_trace(
                go.Scatter(
                    name=upper_regression_key+' regression',
                    x=z_vals_to_evaluate,
                    y=upper_regression_tuple[0],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(color=regression_colors_dict[upper_regression_key], dash = regression_line_styles_dict[upper_regression_key]),
                    showlegend=True
                )
            )
            fig_regressions_dose_quantiles_simple.add_trace(
                go.Scatter(
                    name=lower_regression_key+' regression',
                    x=z_vals_to_evaluate,
                    y=lower_regression_tuple[0],
                    marker=dict(color="#444"),
                    line=dict(color=regression_colors_dict[lower_regression_key], dash = regression_line_styles_dict[lower_regression_key]),
                    mode='lines',
                    fillcolor=fill_color,
                    fill='tonexty',
                    showlegend=True
                )
            )
            
        median_key = "Q50"
        median_dose_regression_tuple = non_parametric_kernel_regressions_dict[median_key]
        fig_regressions_dose_quantiles_simple.add_trace(
            go.Scatter(
                name=median_key+' regression',
                x=z_vals_to_evaluate,
                y=median_dose_regression_tuple[0],
                mode="lines",
                line=dict(color=regression_colors_dict[median_key], dash = regression_line_styles_dict[median_key]),
                showlegend=True
            )
        )
        
        fig_regressions_dose_quantiles_simple.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )
        fig_regressions_dose_quantiles_simple = plotting_funcs.fix_plotly_grid_lines(fig_regressions_dose_quantiles_simple, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + ' - 2d_simple_regressions_quantiles_mean_dose.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig_regressions_dose_quantiles_simple.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + ' - 2d_simple_regressions_quantiles_mean_dose.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig_regressions_dose_quantiles_simple.write_html(html_dose_fig_file_path)



def production_plot_voxelized_axial_dose_distribution_box_violin_by_patient(patient_sp_output_figures_dir_dict,
                                                                            patientUID,
                                                                            pydicom_item,
                                                                            bx_structs,
                                                                            svg_image_scale,
                                                                            svg_image_width,
                                                                            svg_image_height,
                                                                            ):
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        # create box plots of voxelized data
        stats_dose_val_all_MC_trials_voxelized = specific_bx_structure["MC data: voxelized dose results dict (dict of lists)"]
        dose_vals_in_voxel = stats_dose_val_all_MC_trials_voxelized["All dose vals in voxel list"]
        z_range_of_voxel = stats_dose_val_all_MC_trials_voxelized["Voxel z range rounded"]

        max_points_in_voxel = max(len(x) for x in dose_vals_in_voxel)

        dose_output_voxelized_dict_for_pandas_data_frame = {str(z_range_of_voxel[i]): misc_tools.pad_or_truncate(dose_vals_in_voxel[i], max_points_in_voxel) for i in range(len(z_range_of_voxel))}
        dose_output_voxelized_pandas_data_frame = pandas.DataFrame(data=dose_output_voxelized_dict_for_pandas_data_frame)


        #dose_output_voxelized_dict_for_pandas_data_frame = {str(z_range_of_voxel[i]): dose_vals_in_voxel[i] for i in range(len(z_range_of_voxel))}
        #dose_output_voxelized_pandas_data_frame = pandas.DataFrame(data=dose_output_voxelized_dict_for_pandas_data_frame)
        #specific_bx_structure["Output data frames"]["Dose output voxelized"] = dose_output_voxelized_pandas_data_frame

        # box plot
        fig = px.box(dose_output_voxelized_pandas_data_frame, points = False)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )

        svg_dose_fig_name = bx_struct_roi + ' - voxelized_box_plot_dose.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + ' - voxelized_box_plot_dose.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)

        # violin plot
        fig = px.violin(dose_output_voxelized_pandas_data_frame, box=True, points = False)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )

        svg_dose_fig_name = bx_struct_roi + ' - voxelized_violin_plot_dose.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + ' - voxelized_violin_plot_dose.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)




