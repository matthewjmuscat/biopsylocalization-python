import pandas
import plotly.express as px
import plotting_funcs
import numpy as np
import math_funcs as mf
import plotly.graph_objects as go
import misc_tools
import cupy as cp
import cudf

def production_plot_sampled_shift_vector_box_plots_by_patient(patientUID,
                                              patient_sp_output_figures_dir_dict,
                                              structs_referenced_list,
                                              bx_structs,
                                              pydicom_item,
                                              max_simulations,
                                              all_ref_key,
                                              svg_image_scale,
                                              svg_image_width,
                                              svg_image_height,
                                              box_plot_color,
                                              general_plot_name_string):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    structure_name_and_shift_type_dict_for_pandas_data_frame = {'StructureID': None,
                                                                'Structure ref #': None,
                                                                'Struct type': None,
                                                                'Shift X': None,
                                                                'Shift Y': None,
                                                                'Shift Z': None,
                                                                'Shift magnitude': None
                                                                }
    structureID_list = []
    structure_ref_num_list = []
    structure_type_list = []
    shift_vec_x_arr = np.empty((0))
    shift_vec_y_arr = np.empty((0))
    shift_vec_z_arr = np.empty((0))
    shift_vec_mag_arr = np.empty((0))

    for structs in structs_referenced_list:
        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
            structureID = specific_structure["ROI"]
            structureID_list = structureID_list + [structureID]*max_simulations
            structure_reference_number = specific_structure["Ref #"]
            structure_ref_num_list = structure_ref_num_list + [structure_reference_number]*max_simulations
            structure_type_list = structure_type_list + [structs]*max_simulations

            if structs == bx_structs:
                sampled_rigid_shifts = specific_structure["MC data: Total rigid shift vectors arr"]
                sampled_rigid_shifts_magnitudes = np.linalg.norm(sampled_rigid_shifts, axis = 1)
                
            else:
                # create box plots of sampled rigid shifts for each structure                      
                sampled_rigid_shifts = specific_structure['MC data: Generated normal dist random samples arr']
                sampled_rigid_shifts_magnitudes = np.linalg.norm(sampled_rigid_shifts, axis = 1)            

            shift_vec_mag_arr = np.append(shift_vec_mag_arr,sampled_rigid_shifts_magnitudes)
            shift_vec_x_arr = np.append(shift_vec_x_arr, sampled_rigid_shifts[:,0])
            shift_vec_y_arr = np.append(shift_vec_y_arr, sampled_rigid_shifts[:,1])
            shift_vec_z_arr = np.append(shift_vec_z_arr, sampled_rigid_shifts[:,2])

    structure_name_and_shift_type_dict_for_pandas_data_frame['StructureID'] = structureID_list
    structure_name_and_shift_type_dict_for_pandas_data_frame['Structure ref #'] = structure_ref_num_list
    structure_name_and_shift_type_dict_for_pandas_data_frame['Struct type'] = structure_type_list
    structure_name_and_shift_type_dict_for_pandas_data_frame['Shift X'] = shift_vec_x_arr
    structure_name_and_shift_type_dict_for_pandas_data_frame['Shift Y'] = shift_vec_y_arr
    structure_name_and_shift_type_dict_for_pandas_data_frame['Shift Z'] = shift_vec_z_arr
    structure_name_and_shift_type_dict_for_pandas_data_frame['Shift magnitude'] = shift_vec_mag_arr

    structure_name_and_shift_type_dict_pandas_data_frame = pandas.DataFrame(data=structure_name_and_shift_type_dict_for_pandas_data_frame)
    pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["All shift vector magnitudes by structure and shift type"] = structure_name_and_shift_type_dict_pandas_data_frame
    
    
    fig = px.box(structure_name_and_shift_type_dict_pandas_data_frame, x="StructureID", y="Shift magnitude", color="Struct type")

    fig = go.Figure()
    shift_components_to_trace_list = ['Shift magnitude','Shift X', 'Shift Y', 'Shift Z']
    color_by_component_to_trace_list = ['rgba(0, 92, 171, 1)', 'rgba(227, 27, 35,1)', 'rgba(255, 195, 37,1)', 'rgba(0, 200, 255,1)']
    color_by_trace_dict = {key: color_by_component_to_trace_list[i] for i,key in enumerate(shift_components_to_trace_list)}
    for component_to_trace in shift_components_to_trace_list:
        fig.add_trace(go.Box(
            x=structure_name_and_shift_type_dict_pandas_data_frame['StructureID'],
            y=structure_name_and_shift_type_dict_pandas_data_frame[component_to_trace],
            name = component_to_trace,
            marker_color = color_by_trace_dict[component_to_trace]
        ))
    
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Sampled shift (mm)',
        xaxis_title='Structure',
        title='Sampled translation components (' + patientUID +')',
        hovermode="x unified"
    )
    fig.update_layout(
        boxmode='group'
    )


    """
    # box plot
    fig = px.box(structure_name_and_shift_type_dict_pandas_data_frame, points = False)
    fig = fig.update_traces(marker_color = 'rgba(0, 92, 171, 1)') 
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Sampled shift magnitude (mm)',
        xaxis_title='Structure',
        title='Sampled translation magnitudes (' + patientUID +')',
        hovermode="x unified"
    )
    """

    svg_dose_fig_name = patientUID + general_plot_name_string+'.svg'
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = patientUID + general_plot_name_string+'.html'
    html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path)



def production_plot_axial_dose_distribution_all_trials_and_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                                   patientUID,
                                                                   pydicom_item,
                                                                   bx_structs,
                                                                   global_regression_ans,
                                                                   regression_type_ans,
                                                                   parallel_pool,
                                                                   num_bootstraps_for_regression_plots_input,
                                                                   NPKR_bandwidth,
                                                                   svg_image_scale,
                                                                   svg_image_width,
                                                                   svg_image_height,
                                                                   num_z_vals_to_evaluate_for_regression_plots,
                                                                   general_plot_name_string
                                                                   ):
    # generate pandas data frame by reading dose output from file
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]

        dose_output_z_and_radius_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
        pt_radius_bx_coord_sys = dose_output_z_and_radius_dict_for_pandas_data_frame["Radial pos (mm)"]

        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        #bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
        #pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)
        


        # create a 2d scatter plot with all MC trials on plot
        dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"]
        dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_list = dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr.tolist()
        pt_radius_point_wise_for_pd_data_frame_list = []
        axial_Z_point_wise_for_pd_data_frame_list = []
        dose_vals_point_wise_for_pd_data_frame_list = []
        MC_trial_index_point_wise_for_pd_data_frame_list = []
        num_nominal_and_all_MC_trials = dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr.shape[1]
        for pt_index, specific_pt_all_MC_dose_vals in enumerate(dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_list):
            pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + [pt_radius_bx_coord_sys]*num_nominal_and_all_MC_trials
            axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + [bx_points_bx_coords_sys_arr[pt_index,2]]*num_nominal_and_all_MC_trials
            dose_vals_point_wise_for_pd_data_frame_list = dose_vals_point_wise_for_pd_data_frame_list + specific_pt_all_MC_dose_vals
            MC_trial_index_point_wise_for_pd_data_frame_list = MC_trial_index_point_wise_for_pd_data_frame_list + list(range(0,num_nominal_and_all_MC_trials))
        
        # Note that the 0th MC trial num index is the nominal value
        dose_output_dict_by_MC_trial_for_pandas_data_frame = {"Radial pos (mm)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                              "Axial pos Z (mm)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                              "Dose (Gy)": dose_vals_point_wise_for_pd_data_frame_list, 
                                                              "MC trial num": MC_trial_index_point_wise_for_pd_data_frame_list
                                                              }
        
        dose_output_nominal_and_all_MC_trials_pandas_data_frame = pandas.DataFrame.from_dict(data = dose_output_dict_by_MC_trial_for_pandas_data_frame)
        specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"] = dose_output_nominal_and_all_MC_trials_pandas_data_frame

        # do non parametric kernel regression (local linear)
        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=num_z_vals_to_evaluate_for_regression_plots)
        
        if regression_type_ans == True and global_regression_ans == True:
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                parallel_pool,
                x = dose_output_dict_by_MC_trial_for_pandas_data_frame["Axial pos Z (mm)"], 
                y = dose_output_dict_by_MC_trial_for_pandas_data_frame["Dose (Gy)"], 
                eval_x = z_vals_to_evaluate, N=num_bootstraps_for_regression_plots_input, conf_interval=0.95
            )
        elif regression_type_ans == False and global_regression_ans == True:
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                parallel_pool,
                x = dose_output_dict_by_MC_trial_for_pandas_data_frame["Axial pos Z (mm)"], 
                y = dose_output_dict_by_MC_trial_for_pandas_data_frame["Dose (Gy)"], 
                eval_x = z_vals_to_evaluate, N=num_bootstraps_for_regression_plots_input, conf_interval=0.95, bandwidth = NPKR_bandwidth
            )
        elif global_regression_ans == False:
            pass
        
        
        # create 2d scatter dose plot axial (z) vs all doses from all MC trials
        dose_output_all_MC_trials_pandas_data_frame = dose_output_nominal_and_all_MC_trials_pandas_data_frame[dose_output_nominal_and_all_MC_trials_pandas_data_frame["MC trial num"] != 0]
        fig_global = px.scatter(dose_output_all_MC_trials_pandas_data_frame, x="Axial pos Z (mm)", y="Dose (Gy)", color = "MC trial num", width  = svg_image_width, height = svg_image_height)
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
                                
        svg_all_MC_trials_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_dose_fig_name)
        fig_global.write_image(svg_all_MC_trials_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_all_MC_trials_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_dose_fig_name)
        fig_global.write_html(html_all_MC_trials_dose_fig_file_path)
        if global_regression_ans == True:
            svg_all_MC_trials_dose_fig_name = bx_struct_roi + general_plot_name_string+'_with_global_regression.svg'
            svg_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_dose_fig_name)
            fig_regression_only.write_image(svg_all_MC_trials_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_all_MC_trials_dose_fig_name = bx_struct_roi + general_plot_name_string+'_with_global_regression.html'
            html_all_MC_trials_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_dose_fig_name)
            fig_regression_only.write_html(html_all_MC_trials_dose_fig_file_path)
        elif global_regression_ans == False:
            pass
        
def production_3d_scatter_dose_axial_radial_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                   patientUID,
                                                                   pydicom_item,
                                                                   bx_structs,
                                                                   svg_image_scale,
                                                                   svg_image_width,
                                                                   svg_image_height,
                                                                   general_plot_name_string
                                                                   ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        
        dose_output_pandas_data_frame = specific_bx_structure["Output data frames"]["Dose output Z and radius"]
        
        
        
        fig = px.scatter_3d(dose_output_pandas_data_frame, x="Axial pos Z (mm)", y="Radial pos (mm)", z="Mean dose (Gy)", error_z = "STD dose", width  = svg_image_width, height = svg_image_height)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)



def production_2d_scatter_dose_axial_radial_color_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                   patientUID,
                                                                   pydicom_item,
                                                                   bx_structs,
                                                                   svg_image_scale,
                                                                   svg_image_width,
                                                                   svg_image_height,
                                                                   general_plot_name_string
                                                                   ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
         
        dose_output_pandas_data_frame = specific_bx_structure["Output data frames"]["Dose output Z and radius"] 
               
            
        # create 2d scatter dose color map plot axial (z) vs radial (r) vs mean dose (color)
        fig = px.scatter(dose_output_pandas_data_frame, x="Axial pos Z (mm)", y="Radial pos (mm)", color="Mean dose (Gy)", width  = svg_image_width, height = svg_image_height)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)






def production_plot_axial_dose_distribution_quantile_scatter_by_patient(patient_sp_output_figures_dir_dict,
                                                                patientUID,
                                                                pydicom_item,
                                                                bx_structs,
                                                                svg_image_scale,
                                                                svg_image_width,
                                                                svg_image_height,
                                                                general_plot_name_string
                                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]

        dose_output_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
        
        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        #pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

        stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
        mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"].copy()
        std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"].copy()
        quantiles_dose_val_specific_bx_pt_dict_of_lists = stats_dose_val_all_MC_trials_by_bx_pt_list["Quantiles dose by bx pt dict"].copy()
        
          
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

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)



        


        
def production_plot_axial_dose_distribution_quantile_regressions_by_patient(patient_sp_output_figures_dir_dict,
                                                                patientUID,
                                                                pydicom_item,
                                                                bx_structs,
                                                                regression_type_ans,
                                                                parallel_pool,
                                                                NPKR_bandwidth,
                                                                num_bootstraps_for_regression_plots_input,
                                                                svg_image_scale,
                                                                svg_image_width,
                                                                svg_image_height,
                                                                num_z_vals_to_evaluate_for_regression_plots,
                                                                general_plot_name_string
                                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]

        dose_output_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
        pt_radius_bx_coord_sys = dose_output_dict_for_pandas_data_frame["Radial pos (mm)"]

        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        #pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

        stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
        mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"].copy()
        std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"].copy()
        quantiles_dose_val_specific_bx_pt_dict_of_lists = stats_dose_val_all_MC_trials_by_bx_pt_list["Quantiles dose by bx pt dict"].copy()
        
        # Extract nominal values 
        dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"]
        dose_vals_nominal_by_sampled_bx_pt_arr = dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr[:,0]
        dose_vals_nominal_by_sampled_bx_pt_list = dose_vals_nominal_by_sampled_bx_pt_arr.tolist()
        
        
        # do non parametric kernel regression (local linear)
        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=num_z_vals_to_evaluate_for_regression_plots)

        
        # can change the name of this dictionary to 'regression_colors_dict' to make all the regressions a different color
        regression_colors_dict_different = {"Q95":'rgba(255, 0, 0, 1)',
                                    "Q75":'rgba(0, 255, 255, 1)', 
                                    "Q50":'rgba(255, 0, 255, 1)',
                                    'Q25':'rgba(255, 255, 0, 1)',
                                    "Q5":'rgba(0, 0, 255, 1)',
                                    "Mean":'rgba(0, 255, 0, 1)'
        }
        # can change the name of this dictionary to 'regression_colors_dict' to make the paired regressions the same color
        regression_colors_dict = {"Q95":'rgba(0, 0, 0, 1)',
                                    "Q75":'rgba(0, 0, 0, 1)', 
                                    "Q50":'rgba(255, 0, 0, 1)',
                                    'Q25':'rgba(0, 0, 0, 1)',
                                    "Q5":'rgba(0, 0, 0, 1)',
                                    "Mean":'rgba(255, 0, 0, 1)',
                                    "Nominal":'rgba(255, 0, 0, 1)'
        }
        
        regression_line_styles_alternate_dict = {"Q95": '4,14',
                                    "Q75": '10,10,10', 
                                    "Q50": 'dot',
                                    'Q25': '12,2,12',
                                    "Q5": '2,16,2',
                                    "Mean": 'solid'
        }
        
        regression_line_styles_dict = {"Q95": 'dashdot',
                                    "Q75": 'dashdot', 
                                    "Q50": 'longdash',
                                    'Q25': 'dashdot',
                                    "Q5": 'dashdot',
                                    "Mean": 'dot',
                                    "Nominal": 'solid'
        }

        nominal_scatter_marker = 'cross-thin'



        # perform non parametric kernel regression through conditional quantiles and conditional mean doses
        dose_output_dict_for_regression = {"Radial pos (mm)": pt_radius_bx_coord_sys, "Axial pos Z (mm)": bx_points_bx_coords_sys_arr[:,2], "Mean": mean_dose_val_specific_bx_pt, "STD dose": std_dose_val_specific_bx_pt, "Nominal": dose_vals_nominal_by_sampled_bx_pt_list}
        dose_output_dict_for_regression.update(quantiles_dose_val_specific_bx_pt_dict_of_lists)
        non_parametric_kernel_regressions_dict = {}
        data_for_non_parametric_kernel_regressions_dict = {}
        data_keys_to_regress = ["Q95","Q5","Q50","Mean","Q75","Q25","Nominal"]
        #num_bootstraps_mean_and_quantile_data = 15
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
                    N=num_bootstraps_for_regression_plots_input, 
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
                    N=num_bootstraps_for_regression_plots_input, 
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
        fig_regressions_only_quantiles_and_mean.add_trace(
                go.Scatter(
                    name='Nominal',
                    x=dose_output_dict_for_regression["Axial pos Z (mm)"],
                    y=dose_output_dict_for_regression["Nominal"],
                    marker=dict(line_color=regression_colors_dict["Nominal"],
                                symbol = nominal_scatter_marker,
                                line_width = 1,),
                    mode='markers',
                    showlegend=True
                )
            )
            
        
        fig_regressions_only_quantiles_and_mean.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )
        fig_regressions_only_quantiles_and_mean = plotting_funcs.fix_plotly_grid_lines(fig_regressions_only_quantiles_and_mean, y_axis = True, x_axis = True)

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig_regressions_only_quantiles_and_mean.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
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

        mean_key = "Mean"
        mean_dose_regression_tuple = non_parametric_kernel_regressions_dict[mean_key]
        fig_regressions_dose_quantiles_simple.add_trace(
            go.Scatter(
                name=mean_key+' regression',
                x=z_vals_to_evaluate,
                y=mean_dose_regression_tuple[0],
                mode="lines",
                line=dict(color=regression_colors_dict[mean_key], dash = regression_line_styles_dict[mean_key]),
                showlegend=True
            )
        )

        nominal_key = "Nominal"
        nominal_dose_regression_tuple = non_parametric_kernel_regressions_dict[nominal_key]
        fig_regressions_dose_quantiles_simple.add_trace(
            go.Scatter(
                name=nominal_key+' regression',
                x=z_vals_to_evaluate,
                y=nominal_dose_regression_tuple[0],
                mode="lines",
                line=dict(color=regression_colors_dict[nominal_key], dash = regression_line_styles_dict[nominal_key]),
                showlegend=True
            )
        )
        
        
        fig_regressions_dose_quantiles_simple.add_trace(
                go.Scatter(
                    name='Nominal',
                    x=dose_output_dict_for_regression["Axial pos Z (mm)"],
                    y=dose_output_dict_for_regression["Nominal"],
                    marker=dict(line_color=regression_colors_dict["Nominal"],
                                symbol = nominal_scatter_marker,
                                line_width = 1,
                                ),
                    mode='markers',
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

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'_colorwash.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig_regressions_dose_quantiles_simple.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'_colorwash.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig_regressions_dose_quantiles_simple.write_html(html_dose_fig_file_path)







def production_plot_voxelized_axial_dose_distribution_box_plot_by_patient(patient_sp_output_figures_dir_dict,
                                                                            patientUID,
                                                                            pydicom_item,
                                                                            bx_structs,
                                                                            svg_image_scale,
                                                                            svg_image_width,
                                                                            svg_image_height,
                                                                            box_plot_color,
                                                                            general_plot_name_string
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

        specific_bx_structure["Output data frames"]["Dose output voxelized"] = dose_output_voxelized_pandas_data_frame
        specific_bx_structure["Output dicts for data frames"]["Dose output voxelized"] = dose_output_voxelized_dict_for_pandas_data_frame

        # box plot
        fig = px.box(dose_output_voxelized_pandas_data_frame, points = False)
        fig = fig.update_traces(marker_color = box_plot_color)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)

        



def production_plot_voxelized_axial_dose_distribution_violin_plot_by_patient(patient_sp_output_figures_dir_dict,
                                                                            patientUID,
                                                                            pydicom_item,
                                                                            bx_structs,
                                                                            svg_image_scale,
                                                                            svg_image_width,
                                                                            svg_image_height,
                                                                            violin_plot_color,
                                                                            general_plot_name_string
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

        specific_bx_structure["Output data frames"]["Dose output voxelized"] = dose_output_voxelized_pandas_data_frame
        specific_bx_structure["Output dicts for data frames"]["Dose output voxelized"] = dose_output_voxelized_dict_for_pandas_data_frame


        # violin plot
        fig = px.violin(dose_output_voxelized_pandas_data_frame, box=True, points = False)
        fig = fig.update_traces(marker_color = violin_plot_color)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Axial pos Z (mm)',
            title='Dosimetric profile (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)




def production_plot_differential_DVH_showing_N_trials_by_patient(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                num_differential_dvh_plots_to_show,
                                                general_plot_name_string
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        # create differential dvh plots
        differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
        differential_dvh_histogram_percent_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
        differential_dvh_dose_vals_by_MC_trial_1darr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
        differential_dvh_dose_vals_list = differential_dvh_dose_vals_by_MC_trial_1darr.tolist()
        differential_dvh_dose_bins_categorical_list = ['['+str(round(differential_dvh_dose_vals_list[i],1))+','+str(round(differential_dvh_dose_vals_list[i+1],1))+']' for i in range(len(differential_dvh_dose_vals_by_MC_trial_1darr)-1)]
        differential_dvh_histogram_percent_by_MC_trial_list_of_lists = differential_dvh_histogram_percent_by_MC_trial_arr.tolist()
        
        percent_vals_list = []
        dose_bins_list = differential_dvh_dose_bins_categorical_list*num_differential_dvh_plots_to_show 
        mc_trial_index_list = []
        for mc_trial_index in range(num_differential_dvh_plots_to_show):
            percent_vals_list = percent_vals_list + differential_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index]
            mc_trial_index_list = mc_trial_index_list + [mc_trial_index]*len(differential_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index])
        differential_dvh_dict_for_pandas_dataframe = {"Percent volume": percent_vals_list, 
                                                    "Dose (Gy)": dose_bins_list,
                                                    "MC trial": mc_trial_index_list}
        differential_dvh_pandas_dataframe = pandas.DataFrame.from_dict(differential_dvh_dict_for_pandas_dataframe)

        specific_bx_structure["Output data frames"]["Differential DVH by MC trial"] = differential_dvh_pandas_dataframe
        specific_bx_structure["Output dicts for data frames"]["Differential DVH by MC trial"] = differential_dvh_dict_for_pandas_dataframe

        fig_global = px.line(differential_dvh_pandas_dataframe, x="Dose (Gy)", y="Percent volume", color = "MC trial", width  = svg_image_width, height = svg_image_height)
        fig_global.update_layout(
            title = 'Differential DVH of biopsy core (' + patientUID +', '+ bx_struct_roi+'), (Displaying '+str(num_differential_dvh_plots_to_show)+' trials)',
            hovermode = "x unified"
        )
        fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)
        
        svg_differential_dvh_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_differential_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_differential_dvh_fig_name)
        fig_global.write_image(svg_differential_dvh_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_differential_dvh_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_differential_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(html_differential_dvh_fig_name) 
        fig_global.write_html(html_differential_dvh_fig_file_path)


def production_plot_differential_dvh_quantile_box_plot(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                box_plot_color,
                                                nominal_pt_color,
                                                general_plot_name_string
                                                ):
    nominal_scatter_marker = 'circle'
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        # create box plots of differential DVH quantile data                       
        differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
        differential_dvh_histogram_percent_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
        differential_dvh_histogram_percent_by_dose_bin_arr = differential_dvh_histogram_percent_by_MC_trial_arr.T
        differential_dvh_dose_vals_by_MC_trial_1darr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
        differential_dvh_dose_vals_list = differential_dvh_dose_vals_by_MC_trial_1darr.tolist()
        differential_dvh_dose_bins_list = [[round(differential_dvh_dose_vals_list[i],1),round(differential_dvh_dose_vals_list[i+1],1)] for i in range(len(differential_dvh_dose_vals_by_MC_trial_1darr)-1)]

        percent_volume_binned_dict_for_pandas_data_frame = {str(differential_dvh_dose_bins_list[i]): differential_dvh_histogram_percent_by_dose_bin_arr[i,:] for i in range(len(differential_dvh_dose_bins_list))}
        percent_volume_binned_dict_pandas_data_frame = pandas.DataFrame(data=percent_volume_binned_dict_for_pandas_data_frame)

        specific_bx_structure["Output data frames"]["Differential DVH dose binned"] = percent_volume_binned_dict_pandas_data_frame
        specific_bx_structure["Output dicts for data frames"]["Differential DVH dose binned"] = percent_volume_binned_dict_for_pandas_data_frame
        
        """
        # box plot
        fig = px.box(percent_volume_binned_dict_pandas_data_frame, points = False)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Percent volume',
            xaxis_title='Dose (Gy)',
            title='Differential DVH of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )
        """

        differential_dvh_dose_bins_list_strings = [str(i) for i in differential_dvh_dose_bins_list]
        differential_dvh_dose_bins_list_all = differential_dvh_dose_bins_list*differential_dvh_histogram_percent_by_dose_bin_arr.shape[1]
        differential_dvh_dose_bins_list_all_strings = [str(i) for i in differential_dvh_dose_bins_list_all]
        fig_global = go.Figure()
        fig_global.add_trace(
                    go.Box(
                        x=differential_dvh_dose_bins_list_all_strings,
                        y=differential_dvh_histogram_percent_by_dose_bin_arr.flatten('F'),
                        marker_color = box_plot_color,
                        boxpoints=False,
                        name='Differential DVH distribution'
                    )
                )
        fig_global.add_trace(
            go.Scatter(
                name='Nominal',
                x=differential_dvh_dose_bins_list_strings,
                y=differential_dvh_histogram_percent_by_dose_bin_arr[:,0],
                mode="markers",
                marker=dict(line_color = nominal_pt_color,
                                symbol = nominal_scatter_marker
                            ),
                showlegend=True
                )
        )

        fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = False)
        fig_global.update_layout(
            yaxis_title='Percent volume',
            xaxis_title='Dose (Gy)',
            title='Differential DVH of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
            hovermode="x unified"
        )

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig_global.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig_global.write_html(html_dose_fig_file_path)


def production_plot_cumulative_DVH_showing_N_trials_by_patient(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                num_cumulative_dvh_plots_to_show,
                                                general_plot_name_string
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        # create cumulative DVH plots
        cumulative_dvh_dict = specific_bx_structure["MC data: Cumulative DVH dict"]
        cumulative_dvh_histogram_percent_by_MC_trial_arr = cumulative_dvh_dict["Percent arr"]
        cumulative_dvh_dose_vals_by_MC_trial_1darr = cumulative_dvh_dict["Dose vals arr (Gy)"]
        cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists = cumulative_dvh_histogram_percent_by_MC_trial_arr.tolist()
        cumulative_dvh_dose_vals_by_MC_trial_list = cumulative_dvh_dose_vals_by_MC_trial_1darr.tolist()
        percent_vals_list = []
        dose_vals_list = cumulative_dvh_dose_vals_by_MC_trial_list*num_cumulative_dvh_plots_to_show 
        mc_trial_index_list = []
        for mc_trial_index in range(num_cumulative_dvh_plots_to_show):
            percent_vals_list = percent_vals_list + cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index]
            mc_trial_index_list = mc_trial_index_list + [mc_trial_index]*len(cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index])
        cumulative_dvh_dict_for_pandas_dataframe = {"Percent volume": percent_vals_list, 
                                                    "Dose (Gy)": dose_vals_list,
                                                    "MC trial": mc_trial_index_list}
        cumulative_dvh_pandas_dataframe = pandas.DataFrame.from_dict(cumulative_dvh_dict_for_pandas_dataframe)

        specific_bx_structure["Output data frames"]["Cumulative DVH by MC trial"] = cumulative_dvh_pandas_dataframe
        specific_bx_structure["Output dicts for data frames"]["Cumulative DVH by MC trial"] = cumulative_dvh_dict_for_pandas_dataframe

        fig_global = px.line(cumulative_dvh_pandas_dataframe, x="Dose (Gy)", y="Percent volume", color = "MC trial", width  = svg_image_width, height = svg_image_height)
        fig_global.update_layout(
            title='Cumulative DVH of biopsy core (' + patientUID +', '+ bx_struct_roi+'), (Displaying '+str(num_cumulative_dvh_plots_to_show)+' trials)',
            hovermode="x unified"
        )
        fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)
        
        svg_cumulative_dvh_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_cumulative_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_cumulative_dvh_fig_name)
        fig_global.write_image(svg_cumulative_dvh_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_cumulative_dvh_fig_name = bx_struct_roi + general_plot_name_string+'.html'
        html_cumulative_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(html_cumulative_dvh_fig_name) 
        fig_global.write_html(html_cumulative_dvh_fig_file_path)





def production_plot_cumulative_DVH_quantile_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                regression_type_ans,
                                                num_z_vals_to_evaluate_for_regression_plots,
                                                parallel_pool,
                                                NPKR_bandwidth,
                                                num_bootstraps_for_regression_plots_input,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                plot_regression_only_bool,
                                                plot_colorwash_bool,
                                                general_plot_name_string_regression_only,
                                                general_plot_name_string_colorwash
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        cumulative_dvh_dict = specific_bx_structure["MC data: Cumulative DVH dict"]
        cumulative_dvh_dose_vals_by_MC_trial_1darr = cumulative_dvh_dict["Dose vals arr (Gy)"]

        # perform non parametric kernel regression through conditional quantiles and conditional mean cumulative DVH plot
        dose_vals_to_evaluate = np.linspace(min(cumulative_dvh_dose_vals_by_MC_trial_1darr), max(cumulative_dvh_dose_vals_by_MC_trial_1darr), num = num_z_vals_to_evaluate_for_regression_plots)
        quantiles_cumulative_dvh_dict = cumulative_dvh_dict["Quantiles percent dict"]
        cumulative_dvh_percent_by_MC_trial_arr = cumulative_dvh_dict["Percent arr"]
        cumulative_dvh_output_dict_for_regression = {"Dose pts (Gy)": cumulative_dvh_dose_vals_by_MC_trial_1darr, "Nominal": cumulative_dvh_percent_by_MC_trial_arr[0,:]}
        cumulative_dvh_output_dict_for_regression.update(quantiles_cumulative_dvh_dict)
        non_parametric_kernel_regressions_dict = {}
        data_for_non_parametric_kernel_regressions_dict = {}
        data_keys_to_regress = ["Q95","Q5","Q50","Q75","Q25","Nominal"]
        #num_bootstraps_mean_and_quantile_data = 15
        for data_key in data_keys_to_regress:
            data_for_non_parametric_kernel_regressions_dict[data_key]=cumulative_dvh_output_dict_for_regression[data_key].copy()
            
        for data_key, data_to_regress in data_for_non_parametric_kernel_regressions_dict.items():
            if regression_type_ans == True:
                non_parametric_regression_fit, \
                non_parametric_regression_lower, \
                non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = cumulative_dvh_output_dict_for_regression["Dose pts (Gy)"], 
                    y = data_to_regress, 
                    eval_x = dose_vals_to_evaluate, 
                    N=num_bootstraps_for_regression_plots_input, 
                    conf_interval=0.95
                )
            elif regression_type_ans == False:
                non_parametric_regression_fit, \
                non_parametric_regression_lower, \
                non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = cumulative_dvh_output_dict_for_regression["Dose pts (Gy)"], 
                    y = data_to_regress, 
                    eval_x = dose_vals_to_evaluate, 
                    N=num_bootstraps_for_regression_plots_input, 
                    conf_interval=0.95, bandwidth = NPKR_bandwidth
                )
            
            non_parametric_kernel_regressions_dict[data_key] = (
                non_parametric_regression_fit, 
                non_parametric_regression_lower, 
                non_parametric_regression_upper
            )
            
        # create regression figure

        # can change the name of this dictionary to 'regression_colors_dict' to make all the regressions a different color
        regression_colors_dict_different = {"Q95":'rgba(255, 0, 0, 1)',
                                    "Q75":'rgba(0, 255, 255, 1)', 
                                    "Q50":'rgba(255, 0, 255, 1)',
                                    'Q25':'rgba(255, 255, 0, 1)',
                                    "Q5":'rgba(0, 0, 255, 1)',
                                    "Mean":'rgba(0, 255, 0, 1)'
        }
        # can change the name of this dictionary to 'regression_colors_dict' to make the paired regressions the same color
        regression_colors_dict = {"Q95":'rgba(0, 0, 0, 1)',
                                    "Q75":'rgba(0, 0, 0, 1)', 
                                    "Q50":'rgba(255, 0, 0, 1)',
                                    'Q25':'rgba(0, 0, 0, 1)',
                                    "Q5":'rgba(0, 0, 0, 1)',
                                    "Mean":'rgba(255, 0, 0, 1)',
                                    "Nominal":'rgba(255, 0, 0, 1)'
        }
        
        regression_line_styles_alternate_dict = {"Q95": '4,14',
                                    "Q75": '10,10,10', 
                                    "Q50": 'dot',
                                    'Q25': '12,2,12',
                                    "Q5": '2,16,2',
                                    "Mean": 'solid'
        }
        
        regression_line_styles_dict = {"Q95": 'dashdot',
                                    "Q75": 'dashdot', 
                                    "Q50": 'longdash',
                                    'Q25': 'dashdot',
                                    "Q5": 'dashdot',
                                    "Mean": 'dot',
                                    "Nominal": 'solid'
        }

        """
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
        """


        if plot_regression_only_bool == True:
            fig_regressions_only_quantiles_and_mean = go.Figure()
            for data_key,regression_tuple in non_parametric_kernel_regressions_dict.items(): 
                fig_regressions_only_quantiles_and_mean.add_trace(
                    go.Scatter(
                        name=data_key+' regression',
                        x=dose_vals_to_evaluate,
                        y=regression_tuple[0],
                        mode="lines",
                        line=dict(color = regression_colors_dict[data_key], dash = regression_line_styles_dict[data_key]),
                        showlegend=True
                        )
                )
                fig_regressions_only_quantiles_and_mean.add_trace(
                    go.Scatter(
                        name=data_key+': Upper 95% CI',
                        x=dose_vals_to_evaluate,
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
                        x=dose_vals_to_evaluate,
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
                yaxis_title='Percent volume',
                xaxis_title='Dose (Gy)',
                title='Quantile regression of cumulative DVH (' + patientUID +', '+ bx_struct_roi+')',
                hovermode="x unified"
            )
            fig_regressions_only_quantiles_and_mean = plotting_funcs.fix_plotly_grid_lines(fig_regressions_only_quantiles_and_mean, y_axis = True, x_axis = True)

            svg_dose_fig_name = bx_struct_roi + general_plot_name_string_regression_only+'.svg'
            svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
            fig_regressions_only_quantiles_and_mean.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_dose_fig_name = bx_struct_roi + general_plot_name_string_regression_only+'.html'
            html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
            fig_regressions_only_quantiles_and_mean.write_html(html_dose_fig_file_path)

        if plot_colorwash_bool == True:
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
                        x=dose_vals_to_evaluate,
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
                        x=dose_vals_to_evaluate,
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
                    x=dose_vals_to_evaluate,
                    y=median_dose_regression_tuple[0],
                    mode="lines",
                    line=dict(color=regression_colors_dict[median_key], dash = regression_line_styles_dict[median_key]),
                    showlegend=True
                )
            )

            nominal_key = "Nominal"
            median_dose_regression_tuple = non_parametric_kernel_regressions_dict[nominal_key]
            fig_regressions_dose_quantiles_simple.add_trace(
                go.Scatter(
                    name=nominal_key+' regression',
                    x=dose_vals_to_evaluate,
                    y=median_dose_regression_tuple[0],
                    mode="lines",
                    line=dict(color=regression_colors_dict[nominal_key], dash = regression_line_styles_dict[nominal_key]),
                    showlegend=True
                )
            )
            
            fig_regressions_dose_quantiles_simple.update_layout(
                yaxis_title='Percent volume',
                xaxis_title='Dose (Gy)',
                title='Quantile regression of cumulative DVH (' + patientUID +', '+ bx_struct_roi+')',
                hovermode="x unified"
            )
            fig_regressions_dose_quantiles_simple = plotting_funcs.fix_plotly_grid_lines(fig_regressions_dose_quantiles_simple, y_axis = True, x_axis = True)

            svg_dose_fig_name = bx_struct_roi + general_plot_name_string_colorwash+'.svg'
            svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
            fig_regressions_dose_quantiles_simple.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_dose_fig_name = bx_struct_roi + general_plot_name_string_colorwash+'.html'
            html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
            fig_regressions_dose_quantiles_simple.write_html(html_dose_fig_file_path)




def production_plot_containment_probabilities_by_patient(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                regression_type_ans,
                                                parallel_pool,
                                                NPKR_bandwidth,
                                                num_bootstraps_for_regression_plots_input,
                                                num_z_vals_to_evaluate_for_regression_plots,
                                                tissue_class_probability_plot_type_list,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string                                            
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
        bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

        pt_radius_point_wise_for_pd_data_frame_list = []
        axial_Z_point_wise_for_pd_data_frame_list = []
        binom_est_point_wise_for_pd_data_frame_list = []
        total_successes_point_wise_for_pd_data_frame_list = []
        std_err_point_wise_for_pd_data_frame_list = []
        MC_trial_index_point_wise_for_pd_data_frame_list = []
        ROI_name_point_wise_for_pd_data_frame_list = []
        nominal_point_wise_dor_pd_data_frame_list = []
                
        for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
            containment_structure_ROI = containment_structure_key_tuple[0]
            ROI_name_point_wise_for_pd_data_frame_list = ROI_name_point_wise_for_pd_data_frame_list + [containment_structure_ROI]*len(bx_points_bx_coords_sys_arr_list)
            containment_structure_successes_list = containment_structure_dict['Total successes (containment) list']
            containment_structure_binom_est_list = containment_structure_dict["Binomial estimator list"]
            containment_structure_stand_err_list = containment_structure_dict["Standard error (containment) list"]
            containment_structure_nominal_list = containment_structure_dict["Nominal containment list"]
            total_successes_point_wise_for_pd_data_frame_list = total_successes_point_wise_for_pd_data_frame_list + containment_structure_successes_list
            binom_est_point_wise_for_pd_data_frame_list = binom_est_point_wise_for_pd_data_frame_list + containment_structure_binom_est_list
            std_err_point_wise_for_pd_data_frame_list = std_err_point_wise_for_pd_data_frame_list + containment_structure_stand_err_list
            nominal_point_wise_dor_pd_data_frame_list = nominal_point_wise_dor_pd_data_frame_list + containment_structure_nominal_list
            
            pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + pt_radius_bx_coord_sys.tolist()
            axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,2].tolist()     
            
        containment_output_dict_by_MC_trial_for_pandas_data_frame = {"Structure ROI": ROI_name_point_wise_for_pd_data_frame_list, 
                                                                     "Radial pos (mm)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                                     "Axial pos Z (mm)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                     "Mean probability (binom est)": binom_est_point_wise_for_pd_data_frame_list, 
                                                                     "Total successes": total_successes_point_wise_for_pd_data_frame_list, 
                                                                     "STD err": std_err_point_wise_for_pd_data_frame_list,
                                                                     "Nominal containment": nominal_point_wise_dor_pd_data_frame_list
                                                                     }
        
        containment_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_output_dict_by_MC_trial_for_pandas_data_frame)

        specific_bx_structure["Output data frames"]["Containment ouput by bx point"] = containment_output_by_MC_trial_pandas_data_frame
        specific_bx_structure["Output dicts for data frames"]["Containment ouput by bx point"] = containment_output_dict_by_MC_trial_for_pandas_data_frame

        # do non parametric kernel regression (local linear)
        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=num_z_vals_to_evaluate_for_regression_plots)
        all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict = {}
        for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
            containment_structure_ROI = containment_structure_key_tuple[0]
            if regression_type_ans == True:
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit, \
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_lower, \
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Mean probability (binom est)"], 
                    eval_x = z_vals_to_evaluate, N=num_bootstraps_for_regression_plots_input, conf_interval=0.95
                )
            elif regression_type_ans == False:
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit, \
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_lower, \
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Mean probability (binom est)"], 
                    eval_x = z_vals_to_evaluate, N=num_bootstraps_for_regression_plots_input, conf_interval=0.95, bandwidth = NPKR_bandwidth
                )
            containment_regressions_dict = {"Mean regression": all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit, 
                "Lower 95 regression": all_MC_trials_containment_vs_axial_Z_non_parametric_regression_lower, 
                "Upper 95 regression": all_MC_trials_containment_vs_axial_Z_non_parametric_regression_upper}

            all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict[containment_structure_key_tuple] = containment_regressions_dict
        
        # create 2d scatter dose plot axial (z) vs all containment probabilities from all MC trials with regressions
        plot_type_list = tissue_class_probability_plot_type_list
        done_regression_only = False
        nominal_containment_symbols_dict = {0: 'x-thin' , 1: 'circle-open'} # ie. nominal pts that are not contained (0) corresponds to circle-open, and contained (1) corresponds to asterisk               
        for plot_type in plot_type_list:
            # one with error bars on binom est, one without error bars
            if plot_type == 'with_errors':
                fig_global = px.scatter(containment_output_by_MC_trial_pandas_data_frame, 
                                        x="Axial pos Z (mm)", 
                                        y="Mean probability (binom est)", 
                                        color = "Structure ROI", 
                                        symbol = "Nominal containment",
                                        symbol_map = nominal_containment_symbols_dict,
                                        error_y = "STD err", 
                                        #width  = svg_image_width, 
                                        #height = svg_image_height
                                        )
            if plot_type == '':
                fig_global = px.scatter(containment_output_by_MC_trial_pandas_data_frame, 
                                        x="Axial pos Z (mm)", 
                                        y="Mean probability (binom est)", 
                                        color = "Structure ROI",
                                        symbol = "Nominal containment",
                                        symbol_map = nominal_containment_symbols_dict, 
                                        #width  = svg_image_width, 
                                        #height = svg_image_height
                                        )
            
            # marker_line_width = nonzero is necessary in order to see "thin" markers, and must be done by update_traces method
            fig_global = fig_global.update_traces(marker_line_width = 2,
                                                  #marker_line_color = 'black',
                                                  marker_size = 10
                                                  ) 

            # the below for_each_trace method sets the line color of the markers to the marker color, 
            # which was set by the dataframe columns, there is no way to do this directly from the
            # plotly express function call like you can for coloring markers
            fig_global.for_each_trace(lambda trace: trace.update(marker_line_color=trace.marker.color)) 


            # Build dataframe for plotting regression by px.scatter so that we can color by structure

            relative_structure_list = []
            mean_regression_vals_list = []
            z_vals_to_eval_list = []
            for containment_structure_key_tuple, containment_structure_regressions_dict in all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict.items():
                containment_structure_ROI = containment_structure_key_tuple[0]
                sp_structure_mean_regression_list = containment_structure_regressions_dict["Mean regression"].tolist()
                sp_relative_structure_list = [containment_structure_ROI]*len(sp_structure_mean_regression_list)
                mean_regression_vals_list = mean_regression_vals_list + sp_structure_mean_regression_list
                relative_structure_list = relative_structure_list + sp_relative_structure_list
                z_vals_to_eval_list = z_vals_to_eval_list + z_vals_to_evaluate.tolist()

            containment_regression_dictionary_for_pandas_data_frame = {"Mean regression": mean_regression_vals_list,
                                                                       "Z vals to eval": z_vals_to_eval_list,
                                                                       "Relative containment structure": relative_structure_list}
            
            containment_regression_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_regression_dictionary_for_pandas_data_frame)

            fig_regression_only = go.Figure()
            for containment_structure_key_tuple, containment_structure_regressions_dict in all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict.items():
                regression_color = 'rgb'+str(tuple(np.random.randint(low=0,high=225,size=3)))
                containment_structure_ROI = containment_structure_key_tuple[0]
                mean_regression = containment_structure_regressions_dict["Mean regression"]
                lower95_regression = containment_structure_regressions_dict["Lower 95 regression"]
                upper95_regression = containment_structure_regressions_dict["Upper 95 regression"]
                """
                fig_global.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI+' regression',
                        x=z_vals_to_evaluate,
                        y=mean_regression,
                        mode="lines",
                        line=dict(color=regression_color),
                        showlegend=True
                        )
                )
                """
                regression_fig = px.line(containment_regression_pandas_data_frame, 
                                        x="Z vals to eval", 
                                        y="Mean regression", 
                                        color = "Relative containment structure",
                                        #width  = svg_image_width, 
                                        #height = svg_image_height
                                        )
                
                # add all regressions to the global scatter figure, colored by relative structure
                for i in range(0,len(regression_fig.data)):
                    fig_global.add_trace(regression_fig.data[i])
                
                fig_global.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI+' upper 95% CI',
                        x=z_vals_to_evaluate,
                        y=upper95_regression,
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig_global.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI+' lower 95% CI',
                        x=z_vals_to_evaluate,
                        y=lower95_regression,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(0, 100, 20, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    )
                )

                # regressions only figure
                fig_regression_only.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI + ' regression',
                        x=z_vals_to_evaluate,
                        y=mean_regression,
                        mode="lines",
                        line=dict(color=regression_color),
                        showlegend=True
                    )
                )
                fig_regression_only.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI + ' upper 95% CI',
                        x=z_vals_to_evaluate,
                        y=upper95_regression,
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig_regression_only.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI + ' lower 95% CI',
                        x=z_vals_to_evaluate,
                        y=lower95_regression,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(0, 100, 20, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    )
                )

            fig_global.update_layout(
                title='Containment probability (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                hovermode="x unified"
            )
            fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)

            fig_regression_only.update_layout(
                yaxis_title='Conditional mean probability',
                xaxis_title='Axial pos Z (mm)',
                title='Containment probability (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                hovermode="x unified"
            )
            fig_regression_only = plotting_funcs.fix_plotly_grid_lines(fig_regression_only, y_axis = True, x_axis = True)
            
            if plot_type == 'with_errors':
                svg_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter_and_with_errors.svg'
            else:                       
                svg_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter.svg'
            svg_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_containment_fig_name)
            fig_global.write_image(svg_all_MC_trials_containment_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            if plot_type == 'with_errors':
                html_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter_and_with_errors.html'
            else:
                html_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter.html'
            html_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_containment_fig_name)
            fig_global.write_html(html_all_MC_trials_containment_fig_file_path)
            
            if done_regression_only == False:
                svg_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
                svg_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_containment_fig_name)
                fig_regression_only.write_image(svg_all_MC_trials_containment_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                html_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'.html'
                html_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_containment_fig_name)
                fig_regression_only.write_html(html_all_MC_trials_containment_fig_file_path)
            
                done_regression_only = True
            else: 
                pass



def production_plot_mutual_containment_probabilities_by_patient(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                regression_type_ans,
                                                parallel_pool,
                                                NPKR_bandwidth,
                                                num_bootstraps_for_regression_plots_input,
                                                num_z_vals_to_evaluate_for_regression_plots,
                                                tissue_class_probability_plot_type_list,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string,
                                                structure_miss_probability_roi
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
        bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

        

        tumor_tissue_bionomial_est_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue binomial est arr"]
        tumor_tissue_bionomial_se_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue standard error arr"]
        tumor_tissue_conf_int_2d_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue confidence interval 95 arr"]
        tumor_tissue_nominal_containment_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue nominal arr"]
        tumor_tissue_conf_int_lower_arr = tumor_tissue_conf_int_2d_arr[0,:]
        tumor_tissue_conf_int_upper_arr = tumor_tissue_conf_int_2d_arr[1,:]

        pt_radius_point_wise_for_pd_data_frame_list = pt_radius_bx_coord_sys.tolist()
        axial_Z_point_wise_for_pd_data_frame_list = bx_points_bx_coords_sys_arr[:,2].tolist()
        binom_est_point_wise_for_pd_data_frame_list = tumor_tissue_bionomial_est_arr.tolist()
        std_err_point_wise_for_pd_data_frame_list = tumor_tissue_bionomial_se_arr.tolist()
        ROI_name_point_wise_for_pd_data_frame_list = ['DIL']*len(bx_points_bx_coords_sys_arr_list)
        nominal_point_wise_for_pd_data_frame_list = tumor_tissue_nominal_containment_arr.tolist()
        binom_est_lower_CI_point_wise_for_pd_data_frame_list = tumor_tissue_conf_int_lower_arr.tolist()
        binom_est_upper_CI_point_wise_for_pd_data_frame_list = tumor_tissue_conf_int_upper_arr.tolist()
              
        for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
            containment_structure_ROI = containment_structure_key_tuple[0]
            if structure_miss_probability_roi not in containment_structure_ROI:
                continue
            
            ROI_name_point_wise_for_pd_data_frame_list = ROI_name_point_wise_for_pd_data_frame_list + [containment_structure_ROI]*len(bx_points_bx_coords_sys_arr_list)
            containment_structure_binom_est_list = containment_structure_dict["Binomial estimator list"]
            containment_structure_stand_err_list = containment_structure_dict["Standard error (containment) list"]
            containment_structure_CI_list_of_tuples = containment_structure_dict["Confidence interval 95 (containment) list"]
            conf_int_lower_list = [upper_lower_tup[0] for upper_lower_tup in containment_structure_CI_list_of_tuples]
            conf_int_upper_list = [upper_lower_tup[1] for upper_lower_tup in containment_structure_CI_list_of_tuples]
            containment_structure_nominal_list = containment_structure_dict["Nominal containment list"]
            binom_est_point_wise_for_pd_data_frame_list = binom_est_point_wise_for_pd_data_frame_list + containment_structure_binom_est_list
            std_err_point_wise_for_pd_data_frame_list = std_err_point_wise_for_pd_data_frame_list + containment_structure_stand_err_list
            binom_est_lower_CI_point_wise_for_pd_data_frame_list = binom_est_lower_CI_point_wise_for_pd_data_frame_list + conf_int_lower_list
            binom_est_upper_CI_point_wise_for_pd_data_frame_list = binom_est_upper_CI_point_wise_for_pd_data_frame_list + conf_int_upper_list
            nominal_point_wise_for_pd_data_frame_list = nominal_point_wise_for_pd_data_frame_list + containment_structure_nominal_list
            
            pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + pt_radius_bx_coord_sys.tolist()
            axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,2].tolist() 

        # include complemenet of miss structure        
        specific_bx_structure_relative_OAR_dict = specific_bx_structure["MC data: miss structure tissue probability"]
        miss_structure_binom_est_list = specific_bx_structure_relative_OAR_dict["OAR tissue miss binomial est arr"].tolist()
        miss_structure_standard_err_list = specific_bx_structure_relative_OAR_dict["OAR tissue standard error arr"].tolist()
        miss_structure_CI_2d_arr = specific_bx_structure_relative_OAR_dict["OAR tissue miss confidence interval 95 2d arr"]
        miss_structure_CI_lower_list = miss_structure_CI_2d_arr[0,:].tolist()
        miss_structure_CI_upper_list = miss_structure_CI_2d_arr[1,:].tolist()
        miss_structure_nominal_list = specific_bx_structure_relative_OAR_dict["OAR tissue miss nominal arr"].tolist()

        ROI_name_point_wise_for_pd_data_frame_list = ROI_name_point_wise_for_pd_data_frame_list + [structure_miss_probability_roi+' complement']*len(bx_points_bx_coords_sys_arr_list)
        pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + pt_radius_bx_coord_sys.tolist()
        axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,2].tolist() 
        binom_est_point_wise_for_pd_data_frame_list = binom_est_point_wise_for_pd_data_frame_list + miss_structure_binom_est_list
        std_err_point_wise_for_pd_data_frame_list = std_err_point_wise_for_pd_data_frame_list + miss_structure_standard_err_list
        binom_est_lower_CI_point_wise_for_pd_data_frame_list = binom_est_lower_CI_point_wise_for_pd_data_frame_list + miss_structure_CI_lower_list
        binom_est_upper_CI_point_wise_for_pd_data_frame_list = binom_est_upper_CI_point_wise_for_pd_data_frame_list + miss_structure_CI_upper_list
        nominal_point_wise_for_pd_data_frame_list = nominal_point_wise_for_pd_data_frame_list + miss_structure_nominal_list
            
        containment_output_dict_by_MC_trial_for_pandas_data_frame = {"Structure ROI": ROI_name_point_wise_for_pd_data_frame_list, 
                                                                    "Radial pos (mm)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                                    "Axial pos Z (mm)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                    "Mean probability (binom est)": binom_est_point_wise_for_pd_data_frame_list, 
                                                                    "STD err": std_err_point_wise_for_pd_data_frame_list,
                                                                    "Nominal containment": nominal_point_wise_for_pd_data_frame_list,
                                                                    "CI lower vals": binom_est_lower_CI_point_wise_for_pd_data_frame_list,
                                                                    "CI upper vals": binom_est_upper_CI_point_wise_for_pd_data_frame_list
                                                                    }
        
        containment_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_output_dict_by_MC_trial_for_pandas_data_frame)

        specific_bx_structure["Output data frames"]["Mutual containment ouput by bx point"] = containment_output_by_MC_trial_pandas_data_frame
        specific_bx_structure["Output dicts for data frames"]["Mutual containment ouput by bx point"] = containment_output_dict_by_MC_trial_for_pandas_data_frame

        # do non parametric kernel regression (local linear)
        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=num_z_vals_to_evaluate_for_regression_plots)
        containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict = {}
        
        unique_ROIs_list = containment_output_by_MC_trial_pandas_data_frame["Structure ROI"].unique().tolist()

        for roi_to_regress in unique_ROIs_list:
            if regression_type_ans == True:
                miss_structure_probability_vs_axial_Z_NPKR_binom_est_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Mean probability (binom est)"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95
                )
                miss_structure_probability_vs_axial_Z_NPKR_lower_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "CI lower vals"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95
                )
                miss_structure_probability_vs_axial_Z_NPKR_upper_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "CI upper vals"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95
                )
            elif regression_type_ans == False:
                miss_structure_probability_vs_axial_Z_NPKR_binom_est_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Mean probability (binom est)"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95, bandwidth = NPKR_bandwidth
                )
                miss_structure_probability_vs_axial_Z_NPKR_lower_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "CI lower vals"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95, bandwidth = NPKR_bandwidth
                )
                miss_structure_probability_vs_axial_Z_NPKR_upper_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Axial pos Z (mm)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "CI upper vals"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95, bandwidth = NPKR_bandwidth
                )

            containment_regressions_dict = {"Mean regression": miss_structure_probability_vs_axial_Z_NPKR_binom_est_fit, 
                "Lower 95 regression": miss_structure_probability_vs_axial_Z_NPKR_lower_CI_fit, 
                "Upper 95 regression": miss_structure_probability_vs_axial_Z_NPKR_upper_CI_fit
                }

            containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict[roi_to_regress] = containment_regressions_dict

        # create 2d scatter dose plot axial (z) vs all containment probabilities from all MC trials with regressions
        plot_type_list = tissue_class_probability_plot_type_list
        done_regression_only = False
        nominal_containment_symbols_dict = {0: 'x-thin' , 1: 'circle-open'} # ie. nominal pts that are not contained (0) corresponds to circle-open, and contained (1) corresponds to asterisk               
        for plot_type in plot_type_list:
            # one with error bars on binom est, one without error bars
            if plot_type == 'with_errors':
                fig_global = px.scatter(containment_output_by_MC_trial_pandas_data_frame, 
                                        x="Axial pos Z (mm)", 
                                        y="Mean probability (binom est)", 
                                        color = "Structure ROI", 
                                        symbol = "Nominal containment",
                                        symbol_map = nominal_containment_symbols_dict,
                                        error_y = "STD err", 
                                        #width  = svg_image_width, 
                                        #height = svg_image_height
                                        )
            if plot_type == '':
                fig_global = px.scatter(containment_output_by_MC_trial_pandas_data_frame, 
                                        x="Axial pos Z (mm)", 
                                        y="Mean probability (binom est)", 
                                        color = "Structure ROI",
                                        symbol = "Nominal containment",
                                        symbol_map = nominal_containment_symbols_dict, 
                                        #width  = svg_image_width, 
                                        #height = svg_image_height
                                        )
            
            # marker_line_width = nonzero is necessary in order to see "thin" markers, and must be done by update_traces method
            fig_global = fig_global.update_traces(marker_line_width = 2,
                                                  #marker_line_color = 'black',
                                                  marker_size = 10
                                                  ) 

            # the below for_each_trace method sets the line color of the markers to the marker color, 
            # which was set by the dataframe columns, there is no way to do this directly from the
            # plotly express function call like you can for coloring markers
            fig_global.for_each_trace(lambda trace: trace.update(marker_line_color=trace.marker.color)) 


            # Build dataframe for plotting regression by px.scatter so that we can color by structure

            relative_structure_list = []
            mean_regression_vals_list = []
            z_vals_to_eval_list = []
            for containment_structure_ROI, containment_structure_regressions_dict in containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict.items():
                sp_structure_mean_regression_list = containment_structure_regressions_dict["Mean regression"].tolist()
                sp_relative_structure_list = [containment_structure_ROI]*len(sp_structure_mean_regression_list)
                mean_regression_vals_list = mean_regression_vals_list + sp_structure_mean_regression_list
                relative_structure_list = relative_structure_list + sp_relative_structure_list
                z_vals_to_eval_list = z_vals_to_eval_list + z_vals_to_evaluate.tolist()

            containment_regression_dictionary_for_pandas_data_frame = {"Mean regression": mean_regression_vals_list,
                                                                       "Z vals to eval": z_vals_to_eval_list,
                                                                       "Relative containment structure": relative_structure_list}
            
            containment_regression_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_regression_dictionary_for_pandas_data_frame)


            regression_fig = px.line(containment_regression_pandas_data_frame, 
                                        x="Z vals to eval", 
                                        y="Mean regression", 
                                        color = "Relative containment structure",
                                        #width  = svg_image_width, 
                                        #height = svg_image_height
                                        )
                
            # add all regressions to the global scatter figure, colored by relative structure
            for i in range(0,len(regression_fig.data)):
                fig_global.add_trace(regression_fig.data[i])



            fig_regression_only = go.Figure()
            for containment_structure_ROI, containment_structure_regressions_dict in containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict.items():
                regression_color = 'rgb'+str(tuple(np.random.randint(low=0,high=225,size=3)))
                mean_regression = containment_structure_regressions_dict["Mean regression"]
                lower95_regression = containment_structure_regressions_dict["Lower 95 regression"]
                upper95_regression = containment_structure_regressions_dict["Upper 95 regression"]
                """
                fig_global.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI+' regression',
                        x=z_vals_to_evaluate,
                        y=mean_regression,
                        mode="lines",
                        line=dict(color=regression_color),
                        showlegend=True
                        )
                )
                """
                
                
                fig_global.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI+' upper 95% CI',
                        x=z_vals_to_evaluate,
                        y=upper95_regression,
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig_global.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI+' lower 95% CI',
                        x=z_vals_to_evaluate,
                        y=lower95_regression,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(0, 100, 20, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    )
                )

                # regressions only figure
                fig_regression_only.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI + ' regression',
                        x=z_vals_to_evaluate,
                        y=mean_regression,
                        mode="lines",
                        line=dict(color=regression_color),
                        showlegend=True
                    )
                )
                fig_regression_only.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI + ' upper 95% CI',
                        x=z_vals_to_evaluate,
                        y=upper95_regression,
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig_regression_only.add_trace(
                    go.Scatter(
                        name=containment_structure_ROI + ' lower 95% CI',
                        x=z_vals_to_evaluate,
                        y=lower95_regression,
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(0, 100, 20, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    )
                )

            fig_global.update_layout(
                title='Containment probability (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                hovermode="x unified"
            )
            fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)

            fig_regression_only.update_layout(
                yaxis_title='Conditional mean probability',
                xaxis_title='Axial pos Z (mm)',
                title='Containment probability (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                hovermode="x unified"
            )
            fig_regression_only = plotting_funcs.fix_plotly_grid_lines(fig_regression_only, y_axis = True, x_axis = True)
            
            if plot_type == 'with_errors':
                svg_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter_and_with_errors.svg'
            else:                       
                svg_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter.svg'
            svg_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_containment_fig_name)
            fig_global.write_image(svg_all_MC_trials_containment_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            if plot_type == 'with_errors':
                html_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter_and_with_errors.html'
            else:
                html_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'_with_scatter.html'
            html_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_containment_fig_name)
            fig_global.write_html(html_all_MC_trials_containment_fig_file_path)
            
            if done_regression_only == False:
                svg_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
                svg_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_containment_fig_name)
                fig_regression_only.write_image(svg_all_MC_trials_containment_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                html_all_MC_trials_containment_fig_name = bx_struct_roi + general_plot_name_string+'.html'
                html_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_containment_fig_name)
                fig_regression_only.write_html(html_all_MC_trials_containment_fig_file_path)
            
                done_regression_only = True
            else: 
                pass



def production_plot_sobol_indices_global_containment(patient_sp_output_figures_dir_dict,
                                                master_structure_reference_dict,
                                                master_structure_info_dict,
                                                bx_structs,
                                                dil_ref,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                structure_miss_probability_roi,
                                                tissue_class_sobol_global_plot_bool_dict,
                                                box_plot_points_option
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict["Global"]
    fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA: sobol var names by index"]

    num_biopsies = master_structure_info_dict["Global"]["Num biopsies"]
    dataframes_list = [None]*num_biopsies
    for patientUID,pydicom_item in master_structure_reference_dict.items():  
        for bx_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            
            sp_bx_sobol_containment_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol containment dataframe"])
            dataframes_list[bx_index] = sp_bx_sobol_containment_dataframe

    grand_sobol_dataframe = cudf.concat(dataframes_list, ignore_index=True)       

    if tissue_class_sobol_global_plot_bool_dict["Global FO"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO"]

        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_first_order_sobol_list]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all, points = box_plot_points_option)
        fig = fig.update_traces(marker_color = 'rgba(0, 92, 171, 1)') 
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices (tissue classification)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)

    if tissue_class_sobol_global_plot_bool_dict["Global TO"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global TO"]

        column_names_total_order_sobol_list = [name+' TO' for name in fanova_sobol_indices_names_by_index]
        grand_sobol_dataframe_total_order_all = grand_sobol_dataframe[column_names_total_order_sobol_list]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_total_order_all, points = box_plot_points_option)
        fig = fig.update_traces(marker_color = 'rgba(0, 92, 171, 1)') 
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Total order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of total order Sobol indices (tissue classification)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 

    if tissue_class_sobol_global_plot_bool_dict["Global FO, sim vs non sim"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO, sim vs non sim"]

        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        column_names_first_order_sobol_list.append("Simulated bx bool")

        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_first_order_sobol_list]
        color_discrete_map_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all, 
                     points = box_plot_points_option, 
                     color = "Simulated bx bool",
                     color_discrete_map=color_discrete_map_dict)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices by biopsy class (tissue classification)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 


def production_plot_sobol_indices_global_dosimetry(patient_sp_output_figures_dir_dict,
                                                master_structure_reference_dict,
                                                master_structure_info_dict,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                dose_sobol_global_plot_bool_dict,
                                                box_plot_points_option
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict["Global"]
    fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA: sobol var names by index"]
    
    num_biopsies = master_structure_info_dict["Global"]["Num biopsies"]
    dataframes_list = [None]*num_biopsies
    bx_index = 0
    for patientUID,pydicom_item in master_structure_reference_dict.items():  
        for specific_bx_structure in pydicom_item[bx_structs]:
            
            sp_bx_sobol_dose_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol dose dataframe"])
            dataframes_list[bx_index] = sp_bx_sobol_dose_dataframe
            bx_index = bx_index + 1
    del bx_index
    grand_sobol_dataframe = cudf.concat(dataframes_list, ignore_index=True) 

    output_result_key_list = grand_sobol_dataframe["Function output key"].unique().to_arrow().to_pylist()
    output_result_color_list = ['#550527', '#688E26', '#FAA613', '#F44708', '#A10702', '#995FA3']

    color_discrete_map_sim_or_no_sim_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}
    #{for index,key in enumerate(output_result_key_list.keys())}
    #color_discrete_map_output_function_dict = {True: }

    if dose_sobol_global_plot_bool_dict["Global FO"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO"]
        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_first_order_sobol_list]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all, points = box_plot_points_option)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices (dosimetry)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)

    if dose_sobol_global_plot_bool_dict["Global FO by function output"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO by function output"]

        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        column_names_first_order_sobol_list.append("Function output key")
        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_first_order_sobol_list]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all, 
                     points = box_plot_points_option, 
                     color = "Function output key" 
                     )
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices by dosimetry metric (dosimetry)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 

    if dose_sobol_global_plot_bool_dict["Global TO"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global TO"]
        column_names_total_order_sobol_list = [name+' TO' for name in fanova_sobol_indices_names_by_index]
        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_total_order_sobol_list]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all, 
                     points = box_plot_points_option)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Total order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of total order Sobol indices (dosimetry)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)

    if dose_sobol_global_plot_bool_dict["Global TO by function output"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global TO by function output"]

        column_names_total_order_sobol_list = [name+' TO' for name in fanova_sobol_indices_names_by_index]
        column_names_total_order_sobol_list.append("Function output key")
        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_total_order_sobol_list]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all, 
                     points = box_plot_points_option, 
                     color = "Function output key")
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Total order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of total order Sobol indices by dosimetry metric (dosimetry)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)  


    if dose_sobol_global_plot_bool_dict["Global FO, sim only"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO, sim only"]
        sim_bool = True

        grand_sobol_dataframe_simulated_discrim = grand_sobol_dataframe[grand_sobol_dataframe["Simulated bx bool"] == sim_bool]
        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        grand_sobol_dataframe_first_order_all_simulated_discrim = grand_sobol_dataframe_simulated_discrim[column_names_first_order_sobol_list]
        
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all_simulated_discrim, 
                     points = box_plot_points_option)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices (dosimetry) (simulated biopsies)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)

    if dose_sobol_global_plot_bool_dict["Global FO, non-sim only"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO, non-sim only"]
        sim_bool = False
        
        grand_sobol_dataframe_simulated_discrim = grand_sobol_dataframe[grand_sobol_dataframe["Simulated bx bool"] == sim_bool]
        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        grand_sobol_dataframe_first_order_all_simulated_discrim = grand_sobol_dataframe_simulated_discrim[column_names_first_order_sobol_list]

        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all_simulated_discrim, 
                     points = box_plot_points_option)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices (dosimetry) (real biopsies)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 


    if dose_sobol_global_plot_bool_dict["Global FO, sim only by function output"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO, sim only by function output"]
        sim_bool = True
        
        grand_sobol_dataframe_simulated_discrim = grand_sobol_dataframe[grand_sobol_dataframe["Simulated bx bool"] == sim_bool]
        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        column_names_first_order_sobol_list.append("Function output key")
        grand_sobol_dataframe_first_order_all_simulated_discrim = grand_sobol_dataframe_simulated_discrim[column_names_first_order_sobol_list]

        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all_simulated_discrim, 
                     points = box_plot_points_option, 
                     color = "Function output key")
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices by dosimetry metric (dosimetry) (simulated biopsies)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)

    if dose_sobol_global_plot_bool_dict["Global FO, non-sim only by function output"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global FO, non-sim only by function output"]
        sim_bool = False
        
        grand_sobol_dataframe_simulated_discrim = grand_sobol_dataframe[grand_sobol_dataframe["Simulated bx bool"] == sim_bool]
        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        column_names_first_order_sobol_list.append("Function output key")
        grand_sobol_dataframe_first_order_all_simulated_discrim = grand_sobol_dataframe_simulated_discrim[column_names_first_order_sobol_list]

        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all_simulated_discrim, 
                     points = box_plot_points_option, 
                     color = "Function output key")
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices by dosimetry metric (dosimetry) (real biopsies)',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 






def production_plot_patient_cohort(patient_cohort_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    general_plot_name_string,
                                    cohort_output_figures_dir,
                                    box_plot_points_option = 'outliers',
                                    notch_option = True
                                    ):
    
    color_discrete_map_sim_or_no_sim_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}
    fig = go.Figure()

    fig.add_trace(go.Box(
        y = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == False]["Mean probability"],
        x = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == False]["Tissue type"],
        name = 'Actual',
        marker_color = color_discrete_map_sim_or_no_sim_dict[False],
        boxpoints = box_plot_points_option,
        notched = notch_option
    ))
    fig.add_trace(go.Box(
        y = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == True]["Mean probability"],
        x = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == True]["Tissue type"],
        name = 'Simulated',
        marker_color = color_discrete_map_sim_or_no_sim_dict[True],
        boxpoints = box_plot_points_option,
        notched = notch_option
    ))

        
    
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Probability',
        xaxis_title='Tissue classification',
        title='Patient cohort tissue classification probability (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )
    fig.update_layout(
    boxmode='group' # group together boxes of the different traces for each value of x
    )

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 





        
        

