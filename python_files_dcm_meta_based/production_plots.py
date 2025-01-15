import pandas
import plotly.express as px
import plotting_funcs
import numpy as np
import math_funcs as mf
import plotly.graph_objects as go
import misc_tools
import cupy as cp
import cudf
import plotly.figure_factory as ff
from scipy.stats import norm
from scipy import stats
from plotly.subplots import make_subplots
import itertools
from itertools import combinations
import random
import point_containment_tools
import centroid_finder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.kernel_regression import KernelReg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree
import advanced_guidance_map_creator
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import copy
import string
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnnotationBbox, TextArea
import warnings
import dataframe_builders


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
    pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["All MC structure shift vectors"] = structure_name_and_shift_type_dict_pandas_data_frame
    
    
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







def production_plot_sampled_shift_vector_box_plots_by_patient_NEW(patientUID,
                                              patient_sp_output_figures_dir_dict,
                                              pydicom_item,
                                              svg_image_scale,
                                              svg_image_width,
                                              svg_image_height,
                                              general_plot_name_string,
                                              struct_refs_to_include_list,
                                              bx_ref,
                                              bx_sim_types_to_include_list,
                                              all_ref,
                                              num_simulations):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    
    sp_patient_all_structure_shifts_pandas_data_frame = pydicom_item[all_ref]["Multi-structure MC simulation output dataframes dict"]["All MC structure shift vectors"]
    
    df = sp_patient_all_structure_shifts_pandas_data_frame

    # Step 1: Include rows where 'Structure type' is in the list
    condition1 = df['Structure type'].isin(struct_refs_to_include_list)

    # Step 2: Apply specific filtering for biopsies and only include certain biopsy sim types
    condition2 = ((df['Structure type'] != bx_ref) | df["Simulated type"].isin(bx_sim_types_to_include_list))

    # Combine both conditions
    subset_df = df[condition1 & condition2]


    fig = go.Figure()
    shift_components_to_trace_list = ['Shift magnitude','Shift X', 'Shift Y', 'Shift Z']
    color_by_component_to_trace_list = ['rgba(0, 92, 171, 1)', 'rgba(227, 27, 35,1)', 'rgba(255, 195, 37,1)', 'rgba(0, 200, 255,1)']
    color_by_trace_dict = {key: color_by_component_to_trace_list[i] for i,key in enumerate(shift_components_to_trace_list)}
    for component_to_trace in shift_components_to_trace_list:
        fig.add_trace(go.Box(
            x=subset_df['Structure ID'],
            y=subset_df[component_to_trace],
            name = component_to_trace,
            marker_color = color_by_trace_dict[component_to_trace]
        ))

    
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Sampled shift (mm)',
        xaxis_title='Structure',
        title=f'Sampled translation components ({patientUID}, Num simulations: {num_simulations})',
        hovermode="x unified"
    )
    fig.update_layout(
        boxmode='group'
    )

    svg_dose_fig_name = patientUID + general_plot_name_string+'.svg'
    svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = patientUID + general_plot_name_string+'.html'
    html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path)



def production_plot_sampled_shift_vector_box_plots_cohort(cohort_output_figures_dir,
                                                          master_cohort_patient_data_and_dataframes,
                                              svg_image_scale,
                                              svg_image_width,
                                              svg_image_height,
                                              general_plot_name_string,
                                              struct_refs_to_include_list,
                                              bx_ref,
                                              bx_sim_types_to_include_list,
                                              num_simulations,
                                              custom_str_to_append_to_plot_title = ''):
    
    
    df = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: All MC structure shift vectors"]
    
    # Step 1: Include rows where 'Structure type' is in the list
    condition1 = df['Structure type'].isin(struct_refs_to_include_list)

    # Step 2: Apply specific filtering for biopsies and only include certain biopsy sim types
    condition2 = ((df['Structure type'] != bx_ref) | df["Simulated type"].isin(bx_sim_types_to_include_list))

    # Combine both conditions
    subset_df = df[condition1 & condition2]

    fig = go.Figure()
    shift_components_to_trace_list = ['Shift magnitude','Shift X', 'Shift Y', 'Shift Z']
    color_by_component_to_trace_list = ['rgba(0, 92, 171, 1)', 'rgba(227, 27, 35,1)', 'rgba(255, 195, 37,1)', 'rgba(0, 200, 255,1)']
    color_by_trace_dict = {key: color_by_component_to_trace_list[i] for i,key in enumerate(shift_components_to_trace_list)}
    
    # old way where all biopsies will be grouped together regardless of their type
    """
    for component_to_trace in shift_components_to_trace_list:
        fig.add_trace(go.Box(
            x=subset_df['Structure type'],
            y=subset_df[component_to_trace],
            name = component_to_trace,
            marker_color = color_by_trace_dict[component_to_trace]
        ))
    """
    # Add a new column that combines that sim type with structure type:
    #subset_df["struct+sim_type"] = subset_df["Structure type"] + subset_df["Simulated type"]
    subset_df["struct+sim_type"] = subset_df["Structure type"].astype(str) +' (' + subset_df["Simulated type"].astype(str) + ')' # Note that I have to combine them as strings because you cant combine categorical columns with the + operand


    for component_to_trace in shift_components_to_trace_list:
        fig.add_trace(go.Box(
            x=subset_df["struct+sim_type"],
            y=subset_df[component_to_trace],
            name=component_to_trace,
            marker_color=color_by_trace_dict[component_to_trace]
        ))
    
    """
    # Separate non-biopsy structures
    for component_to_trace in shift_components_to_trace_list:
        fig.add_trace(go.Box(
            x=subset_df[subset_df['Structure type'] != bx_ref]['Structure type'],
            y=subset_df[subset_df['Structure type'] != bx_ref][component_to_trace],
            name=component_to_trace,
            marker_color=color_by_trace_dict[component_to_trace]
        ))
    
    # Separate biopsy structures, grouped by "Simulated type"
    biopsy_df = subset_df[subset_df['Structure type'] == bx_ref]

    for simulated_type in biopsy_df["Simulated type"].unique():
        for component_to_trace in shift_components_to_trace_list:
            fig.add_trace(go.Box(
                x=biopsy_df[biopsy_df["Simulated type"] == simulated_type]['Simulated type'],
                y=biopsy_df[biopsy_df["Simulated type"] == simulated_type][component_to_trace],
                name=f"{component_to_trace} ({simulated_type})",  # Include Simulated type in the trace name
                marker_color=color_by_trace_dict[component_to_trace]
            ))
    """


    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Sampled shift (mm)',
        xaxis_title='Structure',
        title=f'Sampled translation components (Cohort {custom_str_to_append_to_plot_title}, Num simulations: {num_simulations})',
        hovermode="x unified"
    )
    fig.update_layout(
        boxmode='group'
    )


    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 












def production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib(pydicom_item,
                                                                                 patientUID,
                                                                                 bx_ref,
                                                                                 all_ref,
                                                                                 value_col_key,
                                                                                 patient_sp_output_figures_dir_dict,
                                                                                 general_plot_name_string,
                                                                                 num_rand_trials_to_show):
    # plotting function
    def plot_quantile_regression_and_more_corrected(df, df_voxelized, sp_patient_all_structure_shifts_pandas_data_frame, patientUID, bx_id, bx_struct_ind, bx_ref):
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))

        # Generate a common x_range for plotting
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Placeholder dictionaries for regression results
        y_regressions = {}

        # Function to perform and plot kernel regression
        def perform_and_plot_kernel_regression(x, y, x_range, label, color, annotation_text = None, target_offset=0):
            kr = KernelReg(endog=y, exog=x, var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            plt.plot(x_range, y_kr, label=label, color=color, linewidth=2)
            
            """
            if annotation_text != None:
                plt.annotate(annotation_text, xy=(x_range[0], y_kr[0]), xytext=(x_range[0], y_kr[0]))
            """
            # Add annotation if provided
            if annotation_text is not None:
                # Determine the total number of points
                total_points = len(x_range)
                
                # Calculate the target index based on the offset, with wrapping
                target_index = (total_points // 5 * target_offset) % total_points
                
                # Target point coordinates
                target_x = x_range[target_index]
                target_y = y_kr[target_index]
                
                # Add annotation with an arrow
                plt.annotate(
                    annotation_text, 
                    xy=(target_x, target_y),  # Point to annotate
                    xytext=(target_x + 1, target_y + 1),  # Offset for text
                    arrowprops=dict(
                        arrowstyle="->",  # Arrow style
                        color=color,      # Arrow color
                        lw=1.5            # Line width
                    ),
                    fontsize=10,        # Font size of annotation
                    color=color,        # Color of annotation text
                    bbox=dict(
                        boxstyle="round,pad=0.3",  # Text box style
                        edgecolor=color,          # Edge color of box
                        facecolor="white",        # Background color of box
                        alpha=0.8                 # Transparency of box
                    )
                )

        # Perform kernel regression for each quantile and store the y-values
        for quantile in [0.05, 0.25, 0.75, 0.95]:
            q_df = df.groupby('Z (Bx frame)')[value_col_key].quantile(quantile).reset_index()
            kr = KernelReg(endog=q_df[value_col_key], exog=q_df['Z (Bx frame)'], var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            y_regressions[quantile] = y_kr

        # Filling the space between the quantile lines
        plt.fill_between(x_range, y_regressions[0.05], y_regressions[0.25], color='springgreen', alpha=1)
        plt.fill_between(x_range, y_regressions[0.25], y_regressions[0.75], color='dodgerblue', alpha=1)
        plt.fill_between(x_range, y_regressions[0.75], y_regressions[0.95], color='springgreen', alpha=1)
        
        # Additional plot enhancements
        # Plot line for MC trial num = 0
        # Kernel regression for MC trial num = 0 subset
        
        mc_trial_0 = df[df['MC trial num'] == 0]
        perform_and_plot_kernel_regression(mc_trial_0['Z (Bx frame)'], mc_trial_0[value_col_key], x_range, 'Nominal', 'red')
        

        # KDE and mean dose per Original pt index
        kde_max_doses = []
        mean_doses = []
        z_vals = []
        for z_val in df['Z (Bx frame)'].unique():
            pt_data = df[df['Z (Bx frame)'] == z_val]
            kde = gaussian_kde(pt_data[value_col_key])
            kde_doses = np.linspace(pt_data[value_col_key].min(), pt_data[value_col_key].max(), 500)
            max_density_dose = kde_doses[np.argmax(kde(kde_doses))]
            kde_max_doses.append(max_density_dose)
            mean_doses.append(pt_data[value_col_key].mean())
            z_vals.append(z_val)
        
        perform_and_plot_kernel_regression(z_vals, kde_max_doses, x_range, 'KDE Max Density Dose', 'magenta')
        perform_and_plot_kernel_regression(z_vals, mean_doses, x_range, 'Mean Dose', 'orange')

        

        # Line plot for each trial
        """
        num_mc_trials_plus_nom = df_voxelized['MC trial num'].nunique()
        for trial in range(1,num_mc_trials_plus_nom):
            df_sp_trial = df[df["MC trial num"] == trial].sort_values(by='Z (Bx frame)') # sorting is to make sure that the lines are drawn properly
            df_z_simple = df_sp_trial.drop_duplicates(subset=['Z (Bx frame)'], keep='first') # remove points that have the same z value so that the line plots look better
            #plt.plot(df_z_simple['Z (Bx frame)'], df_z_simple[value_col_key], color='grey', alpha=0.1, linewidth=1, zorder = 0.9)  # 'linewidth' controls the thickness of the line, zorder puts these lines below the fill betweens!
            plt.scatter(
                df_z_simple['Z (Bx frame)'], 
                df_z_simple[value_col_key], 
                color='grey', 
                alpha=0.1, 
                s=10,  # Size of dots, adjust as needed
                zorder=0.9
            )
        """

        ## Instead want to show regressions of random trials so that we can appreciate structure
        annotation_offset_index = 0
        for trial in range(1,num_rand_trials_to_show + 1): # +1 because we start at 1 in range()
            mc_trial_shift_vec_df = sp_patient_all_structure_shifts_pandas_data_frame[(sp_patient_all_structure_shifts_pandas_data_frame["Trial"] == trial) &
                                                                                   (sp_patient_all_structure_shifts_pandas_data_frame["Structure type"] == bx_ref) & 
                                                                                   (sp_patient_all_structure_shifts_pandas_data_frame["Structure index"] == bx_struct_ind)].reset_index(drop=True)
            mc_trial = df[df['MC trial num'] == trial]
            mc_trial_voxelized = df_voxelized[df_voxelized['MC trial num'] == trial]

            

            x_dist = mc_trial_shift_vec_df.at[0,'Shift X']
            y_dist = mc_trial_shift_vec_df.at[0,'Shift Y']
            z_dist = mc_trial_shift_vec_df.at[0,'Shift Z']
            d_tot = mc_trial_shift_vec_df.at[0,'Shift magnitude']

            annotation_text_for_trial = f"({x_dist:.1f},{y_dist:.1f},{z_dist:.1f}), d = {d_tot:.1f}"
            
            perform_and_plot_kernel_regression(mc_trial['Z (Bx frame)'], mc_trial[value_col_key], x_range, f"Trial: {trial}", 'gray', annotation_text = annotation_text_for_trial, target_offset=annotation_offset_index)
            
            plt.scatter(
                mc_trial_voxelized['Z (Bx frame)'], 
                mc_trial_voxelized[value_col_key], 
                color='grey', 
                alpha=0.1, 
                s=10,  # Size of dots, adjust as needed
                zorder=1.1
            )
            annotation_offset_index += 1
        """
        for trial in range(1,num_rand_trials_to_show):
            df_sp_trial = df_voxelized[df_voxelized["MC trial num"] == trial]
            plt.plot(df_sp_trial['Z (Bx frame)'], df_sp_trial[value_col_key], color='grey', alpha=0.1, linewidth=1, zorder = 1.1)  # 'linewidth' controls the thickness of the line, zorder puts these lines below the fill betweens!
        """


        plt.title(f'Quantile Regression with Filled Areas Between Lines - {patientUID} - {bx_id}')
        plt.xlabel('Z (Bx frame)')
        plt.ylabel(value_col_key)
        plt.legend(['5th-25th Percentile', '25th-75th Percentile', '75th-95th Percentile', 'Nominal', 'Max density dose', 'Mean dose'], loc='best', facecolor = 'white')
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        return fig
    
    # plotting loop
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]

    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
        bx_struct_roi = specific_bx_structure["ROI"]
        bx_struct_ind = specific_bx_structure["Index number"]

        sp_patient_all_structure_shifts_pandas_data_frame = pydicom_item[all_ref]["Multi-structure MC simulation output dataframes dict"]["All MC structure shift vectors"]

        
        dose_output_nominal_and_all_MC_trials_pandas_data_frame = specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"]

        dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame = specific_bx_structure["Output data frames"]["Voxel-wise dose output by MC trial number"]

        fig = plot_quantile_regression_and_more_corrected(dose_output_nominal_and_all_MC_trials_pandas_data_frame,
                                                          dose_output_nominal_and_all_MC_trials_fully_voxelized_pandas_data_frame,
                                                          sp_patient_all_structure_shifts_pandas_data_frame,
                                                          patientUID, 
                                                          bx_struct_roi,
                                                          bx_struct_ind,
                                                          bx_ref)

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

        fig.savefig(svg_dose_fig_file_path, format='svg')

        # clean up for memory
        plt.close(fig)
            



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

        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
        """
        dose_output_z_and_radius_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
        pt_radius_bx_coord_sys = dose_output_z_and_radius_dict_for_pandas_data_frame["R (Bx frame)"]

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
        dose_output_dict_by_MC_trial_for_pandas_data_frame = {"R (Bx frame)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                              "Z (Bx frame)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                              "Dose (Gy)": dose_vals_point_wise_for_pd_data_frame_list, 
                                                              "MC trial num": MC_trial_index_point_wise_for_pd_data_frame_list
                                                              }
        
        
        dose_output_nominal_and_all_MC_trials_pandas_data_frame = pandas.DataFrame.from_dict(data = dose_output_dict_by_MC_trial_for_pandas_data_frame)
        """
        #dose_output_dict_by_MC_trial_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Point-wise dose output by MC trial number"]
        dose_output_nominal_and_all_MC_trials_pandas_data_frame = specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"]

        # do non parametric kernel regression (local linear)
        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=num_z_vals_to_evaluate_for_regression_plots)
        
        if regression_type_ans == True and global_regression_ans == True:
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                parallel_pool,
                x = dose_output_dict_by_MC_trial_for_pandas_data_frame["Z (Bx frame)"], 
                y = dose_output_dict_by_MC_trial_for_pandas_data_frame["Dose (Gy)"], 
                eval_x = z_vals_to_evaluate, N=num_bootstraps_for_regression_plots_input, conf_interval=0.95
            )
        elif regression_type_ans == False and global_regression_ans == True:
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_fit, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_lower, \
            all_MC_trials_dose_vs_axial_Z_non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                parallel_pool,
                x = dose_output_dict_by_MC_trial_for_pandas_data_frame["Z (Bx frame)"], 
                y = dose_output_dict_by_MC_trial_for_pandas_data_frame["Dose (Gy)"], 
                eval_x = z_vals_to_evaluate, N=num_bootstraps_for_regression_plots_input, conf_interval=0.95, bandwidth = NPKR_bandwidth
            )
        elif global_regression_ans == False:
            pass
        
        
        # create 2d scatter dose plot axial (z) vs all doses from all MC trials
        dose_output_all_MC_trials_pandas_data_frame = dose_output_nominal_and_all_MC_trials_pandas_data_frame[dose_output_nominal_and_all_MC_trials_pandas_data_frame["MC trial num"] != 0]
        fig_global = px.scatter(dose_output_all_MC_trials_pandas_data_frame, x="Z (Bx frame)", y="Dose (Gy)", color = "MC trial num", width  = svg_image_width, height = svg_image_height)
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
                xaxis_title='Z (Bx frame)',
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
        
        
        
        fig = px.scatter_3d(dose_output_pandas_data_frame, x="Z (Bx frame)", y="R (Bx frame)", z="Mean dose (Gy)", error_z = "STD dose", width  = svg_image_width, height = svg_image_height)
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
        fig = px.scatter(dose_output_pandas_data_frame, x="Z (Bx frame)", y="R (Bx frame)", color="Mean dose (Gy)", width  = svg_image_width, height = svg_image_height)
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

        #dose_output_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
        
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
                x=dose_output_dict_for_pandas_data_frame["Z (Bx frame)"],
                y=dose_output_dict_for_pandas_data_frame["Mean dose (Gy)"],
                mode='markers',
                marker_color=regression_colors_dict["Mean"],
                showlegend=True
            ),
            go.Scatter(
                name='95th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Z (Bx frame)"],
                y=dose_output_dict_for_pandas_data_frame["Q95"],
                mode='markers',
                marker_color=regression_colors_dict["Q95"],
                showlegend=True
            ),
            go.Scatter(
                name='75th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Z (Bx frame)"],
                y=dose_output_dict_for_pandas_data_frame["Q75"],
                mode='markers',
                marker_color=regression_colors_dict["Q75"],
                showlegend=True
            ),
            go.Scatter(
                name='50th Quantile (median)',
                x=dose_output_dict_for_pandas_data_frame["Z (Bx frame)"],
                y=dose_output_dict_for_pandas_data_frame["Q50"],
                mode='markers',
                marker_color=regression_colors_dict["Q50"],
                showlegend=True
            ),
            go.Scatter(
                name='25th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Z (Bx frame)"],
                y=dose_output_dict_for_pandas_data_frame["Q25"],
                mode='markers',
                marker_color=regression_colors_dict["Q25"],
                showlegend=True
            ),
            go.Scatter(
                name='5th Quantile',
                x=dose_output_dict_for_pandas_data_frame["Z (Bx frame)"],
                y=dose_output_dict_for_pandas_data_frame["Q5"],
                mode='markers',
                marker_color=regression_colors_dict["Q5"],
                showlegend=True
            )
        ])
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Z (Bx frame)',
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

        #dose_output_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
        pt_radius_bx_coord_sys = dose_output_dict_for_pandas_data_frame["R (Bx frame)"]

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
        dose_output_dict_for_regression = {"R (Bx frame)": pt_radius_bx_coord_sys, "Z (Bx frame)": bx_points_bx_coords_sys_arr[:,2], "Mean": mean_dose_val_specific_bx_pt, "STD dose": std_dose_val_specific_bx_pt, "Nominal": dose_vals_nominal_by_sampled_bx_pt_list}
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
                    x = dose_output_dict_for_regression["Z (Bx frame)"], 
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
                    x = dose_output_dict_for_regression["Z (Bx frame)"], 
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
                    x=dose_output_dict_for_regression["Z (Bx frame)"],
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
            xaxis_title='Z (Bx frame)',
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
                    x=dose_output_dict_for_regression["Z (Bx frame)"],
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
            xaxis_title='Z (Bx frame)',
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

        dose_output_voxelized_pandas_data_frame = specific_bx_structure["Output data frames"]["Dose output voxelized"]

        # box plot
        fig = px.box(dose_output_voxelized_pandas_data_frame, points = False)
        fig = fig.update_traces(marker_color = box_plot_color)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Z (Bx frame)',
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

        dose_output_voxelized_pandas_data_frame = specific_bx_structure["Output data frames"]["Dose output voxelized"]

        # violin plot
        fig = px.violin(dose_output_voxelized_pandas_data_frame, box=True, points = False)
        fig = fig.update_traces(marker_color = violin_plot_color)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Dose (Gy)',
            xaxis_title='Z (Bx frame)',
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
        
        differential_dvh_pandas_dataframe = specific_bx_structure["Output data frames"]["Differential DVH by MC trial"]
        differential_dvh_pandas_dataframe_N_trials = differential_dvh_pandas_dataframe[differential_dvh_pandas_dataframe["MC trial"] < num_differential_dvh_plots_to_show]

        fig_global = px.line(differential_dvh_pandas_dataframe_N_trials, x="Dose (Gy)", y="Percent volume", color = "MC trial", width  = svg_image_width, height = svg_image_height)
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

        #specific_bx_structure["Output data frames"]["Differential DVH dose binned"] = percent_volume_binned_dict_pandas_data_frame
        #specific_bx_structure["Output dicts for data frames"]["Differential DVH dose binned"] = percent_volume_binned_dict_for_pandas_data_frame
        
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




def production_plot_differential_dvh_quantile_plot_NEW(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                general_plot_name_string,
                                                important_info,
                                                live_display
                                                ):
    plt.ioff()

    def plot_kernel_regression(x, y, label, color):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                kr = KernelReg(endog=y, exog=x, var_type='c')
                x_range = np.linspace(x.min(), x.max(), 500)  # Ensure this range aligns with your x-axis
                y_kr, _ = kr.fit(x_range)
                plt.plot(x_range, y_kr, label=label, color=color)
                if w:
                    for warning in w:
                        important_info.add_text_line(f"Warning encountered for differential DVH nominal - Patient: {patientUID}, Bx ID: {bx_struct_roi}: {warning}", live_display)
        except np.linalg.LinAlgError:
            important_info.add_text_line("SVD did not converge for differential DVH nominal - Patient: {patientUID}, Bx ID: {bx_struct_roi}", live_display)
            plt.plot(x, y, label=label, color=color, linestyle='-', marker=None)

            
    def plot_filled_quantiles(df, x_col, y_col, patientUID, bx_struct_roi):
        fig = plt.figure(figsize=(10, 6))  # Adjust size as needed

        # Calculate and plot kernel regressions for the desired quantiles
        quantiles = [0.05, 0.25, 0.75, 0.95]
        quantile_dfs = {}
        x_ranges = {}
        y_krs = {}
        
        for q in quantiles:
            q_df = df.groupby(x_col)[y_col].quantile(q).reset_index()
            quantile_dfs[q] = q_df
            x_range = np.linspace(df[x_col].min(), df[x_col].max(), 500)
            x_ranges[q] = x_range

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    kr = KernelReg(endog=q_df[y_col], exog=q_df[x_col], var_type='c')
                    y_kr, _ = kr.fit(x_range)
                    y_krs[q] = y_kr
                    if w:
                        for warning in w:
                            important_info.add_text_line(f"Warning encountered for differential DVH quantile {q} - Patient: {patientUID}, Bx ID: {bx_struct_roi}: {warning}", live_display)

            except np.linalg.LinAlgError:
                important_info.add_text_line(f"SVD did not converge for differential DVH quantile {q} - Patient: {patientUID}, Bx ID: {bx_struct_roi}", live_display)
                # Perform linear interpolation
                if not q_df.empty:
                    y_kr = np.interp(x_range, q_df[x_col], q_df[y_col])
                else:
                    y_kr = np.zeros_like(x_range)
                y_krs[q] = y_kr  # Use interpolated values as a fallback or all zeros if have an empty dataframe, which we should never have
            

        # Filling the areas between quantile regressions
        plt.fill_between(x_ranges[0.05], y_krs[0.05], y_krs[0.25], color='springgreen', alpha=1)
        plt.fill_between(x_ranges[0.25], y_krs[0.25], y_krs[0.75], color='dodgerblue', alpha=1)
        plt.fill_between(x_ranges[0.75], y_krs[0.75], y_krs[0.95], color='springgreen', alpha=1)

        # 3. Kernel regression for 'MC trial' == 0
        df_trial_0 = df[df['MC trial'] == 0]
        plot_kernel_regression(df_trial_0[x_col], df_trial_0[y_col], 'Nominal', 'red')

        # Scatter plot for the data points
        #plt.scatter(df[x_col], df[y_col], color='grey', alpha=0.1, s=10)  # 's' controls size, 'alpha' controls transparency
        # Line plot for the data points
        #plt.plot(df[x_col], df[y_col], color='grey', alpha=0.1, linewidth=1)  # 'linewidth' controls the thickness of the line

        num_mc_trials_plus_nom = df['MC trial'].nunique()

        # Line plot for each trial
        for trial in range(1,num_mc_trials_plus_nom):
            df_sp_trial = df[df["MC trial"] == trial].sort_values(by=x_col) # sorting is to make sure that the lines are drawn properly
            plt.plot(df_sp_trial[x_col], df_sp_trial[y_col], color='grey', alpha=0.1, linewidth=1, zorder = 0.9)  # 'linewidth' controls the thickness of the line, zorder puts these lines below the fill betweens!

        plt.title(f'Differential DVH Quantile Regression - {patientUID} - {bx_struct_roi}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        #handles, labels = plt.gca().get_legend_handles_labels()
        #plt.legend(handles[:1], labels[:1], title="Legend", frameon=True, facecolor='white')
        plt.legend(['5th-25th Percentile', '25th-75th Percentile', '75th-95th Percentile', 'Nominal'], loc='best', facecolor = 'white')
        plt.tight_layout()

        return fig

    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]

    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        differential_dvh_pandas_dataframe = specific_bx_structure["Output data frames"]["Differential DVH by MC trial"]

        df = differential_dvh_pandas_dataframe

        fig = plot_filled_quantiles(df, 'Dose bin center (Gy)', 'Percent volume', patientUID, bx_struct_roi)
        
        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

        fig.savefig(svg_dose_fig_file_path, format='svg')

        # clean up for memory
        plt.close(fig)



def production_plot_differential_dvh_violin_NEW(patient_sp_output_figures_dir_dict,
                                                pydicom_item,
                                                patientUID,
                                                bx_structs,
                                                general_plot_name_string):
    nominal_scatter_marker = 'circle'
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]

        differential_dvh_pandas_dataframe = specific_bx_structure["Output data frames"]["Differential DVH by MC trial"]
        differential_dvh_pandas_dataframe_nominal = differential_dvh_pandas_dataframe[differential_dvh_pandas_dataframe["MC trial"] == 0]
        
        sns.violinplot(x="Dose bin (Gy)", 
                        y="Percent volume",
                        data=differential_dvh_pandas_dataframe, 
                        palette="Set3")


        # 's' is the size of the points, and 'color' specifies the color of the points
        plt.scatter(x=["Dose bin (Gy)"] * len(differential_dvh_pandas_dataframe_nominal), 
                    y=differential_dvh_pandas_dataframe_nominal["Percent volume"], 
                    color='red', 
                    s=100, 
                    edgecolor='k', 
                    alpha=0.7, 
                    label='Nominal')

        # Adding titles and labels for clarity
        plt.title('Violin Plot differential DVH - ' + bx_struct_roi)
        plt.xlabel('Dose bin (Gy)')
        plt.ylabel('Percent volume')
        plt.legend()
        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

        plt.savefig(svg_dose_fig_file_path, format='svg')



def production_plot_cumulative_DVH_kernel_quantile_regression_NEW(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_ref,
                                                num_cumulative_dvh_plots_to_show,
                                                general_plot_name_string):
    
    def plot_quantile_regression(df, num_samples_to_show = 50):
        fig = plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")


        # Plotting all lines in grey with low alpha, except for MC trial == 0 in red
        # Plot each MC trial as a separate line, except for MC trial == 0
        unique_trials = df['MC trial'].unique()
        if len(unique_trials) > num_samples_to_show:
            selected_trials = random.sample(list(unique_trials[unique_trials != 0]), num_samples_to_show - 1) + [0]
        else:
            selected_trials = unique_trials

        for trial in selected_trials:
            if trial == 0:
                sns.lineplot(x="Dose (Gy)", y="Percent volume", data=df[df['MC trial'] == trial], color='red', alpha=1)
            else:
                sns.lineplot(x="Dose (Gy)", y="Percent volume", data=df[df['MC trial'] == trial], color='grey', alpha=0.01, legend=None)


        # Corrected quantile regression and filling
        quantiles = [5, 25, 75, 95]
        quantile_values = {q: [] for q in quantiles}
        x_vals = np.linspace(df["Dose (Gy)"].min(), df["Dose (Gy)"].max(), 100)

        # Computing quantiles for each x value
        for x in x_vals:
            # Find rows where Dose (Gy) is close to x to mitigate exact match issue
            dose_df = df[np.abs(df['Dose (Gy)'] - x) < (df['Dose (Gy)'].max() - df['Dose (Gy)'].min()) / 100]
            for q in quantiles:
                quantile_values[q].append(dose_df['Percent volume'].quantile(q/100))

        # Plotting filled areas between quantiles with correct colors
        plt.fill_between(x_vals, quantile_values[5], quantile_values[25], color='green', alpha=0.7, label='5th-25th Quantile')
        plt.fill_between(x_vals, quantile_values[25], quantile_values[75], color='blue', alpha=0.7, label='25th-75th Quantile')
        plt.fill_between(x_vals, quantile_values[75], quantile_values[95], color='green', alpha=0.7, label='75th-95th Quantile')

        plt.xlabel("Dose (Gy)")
        plt.ylabel("Percent volume")
        plt.title(f"Dose vs. Percent Volume by MC Trial with Quantile Regressions for ROI {bx_struct_roi}")
        plt.legend()
        return fig

    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
        bx_struct_roi = specific_bx_structure["ROI"]

        cumulative_dvh_pandas_dataframe = specific_bx_structure["Output data frames"]["Cumulative DVH by MC trial"] 
        df = cumulative_dvh_pandas_dataframe
        fig = plot_quantile_regression(df, num_samples_to_show = num_cumulative_dvh_plots_to_show)

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

        fig.savefig(svg_dose_fig_file_path, format='svg')

        # clean up for memory
        plt.close(fig)
                



def production_plot_cumulative_DVH_kernel_quantile_regression_NEW_v2(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                general_plot_name_string,
                                                important_info,
                                                live_display
                                                ):
    plt.ioff()

    def plot_kernel_regression(x, y, label, color):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                kr = KernelReg(endog=y, exog=x, var_type='c')
                x_range = np.linspace(x.min(), x.max(), 500)  # Ensure this range aligns with your x-axis
                y_kr, _ = kr.fit(x_range)
                plt.plot(x_range, y_kr, label=label, color=color)
                if w:
                    for warning in w:
                        important_info.add_text_line(f"Warning encountered for differential DVH nominal - Patient: {patientUID}, Bx ID: {bx_struct_roi}: {warning}", live_display)
        except np.linalg.LinAlgError:
            important_info.add_text_line("SVD did not converge for differential DVH nominal - Patient: {patientUID}, Bx ID: {bx_struct_roi}", live_display)
            plt.plot(x, y, label=label, color=color, linestyle='-', marker=None)
            
    def plot_filled_quantiles(df, x_col, y_col, patientUID, bx_struct_roi):
        fig = plt.figure(figsize=(10, 6))  # Adjust size as needed

        # Calculate and plot kernel regressions for the desired quantiles
        quantiles = [0.05, 0.25, 0.75, 0.95]
        quantile_dfs = {}
        x_ranges = {}
        y_krs = {}
        
        for q in quantiles:
            q_df = df.groupby(x_col)[y_col].quantile(q).reset_index()
            quantile_dfs[q] = q_df
            x_range = np.linspace(df[x_col].min(), df[x_col].max(), 500)
            x_ranges[q] = x_range

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    kr = KernelReg(endog=q_df[y_col], exog=q_df[x_col], var_type='c')
                    y_kr, _ = kr.fit(x_range)
                    y_krs[q] = y_kr
                    if w:
                        for warning in w:
                            important_info.add_text_line(f"Warning encountered for differential DVH quantile {q} - Patient: {patientUID}, Bx ID: {bx_struct_roi}: {warning}", live_display)
            
            except np.linalg.LinAlgError:
                important_info.add_text_line(f"SVD did not converge for differential DVH quantile {q} - Patient: {patientUID}, Bx ID: {bx_struct_roi}", live_display)
                # Perform linear interpolation
                if not q_df.empty:
                    y_kr = np.interp(x_range, q_df[x_col], q_df[y_col])
                else:
                    y_kr = np.zeros_like(x_range)
                y_krs[q] = y_kr  # Use interpolated values as a fallback or all zeros if have an empty dataframe, which we should never have
            

        # Filling the areas between quantile regressions
        plt.fill_between(x_ranges[0.05], y_krs[0.05], y_krs[0.25], color='springgreen', alpha=1)
        plt.fill_between(x_ranges[0.25], y_krs[0.25], y_krs[0.75], color='dodgerblue', alpha=1)
        plt.fill_between(x_ranges[0.75], y_krs[0.75], y_krs[0.95], color='springgreen', alpha=1)

        # 3. Kernel regression for 'MC trial' == 0
        df_trial_0 = df[df['MC trial'] == 0]
        plot_kernel_regression(df_trial_0[x_col], df_trial_0[y_col], 'Nominal', 'red')

        # Scatter plot for the data points
        #plt.scatter(df[x_col], df[y_col], color='grey', alpha=0.1, s=10)  # 's' controls size, 'alpha' controls transparency
        
        num_mc_trials_plus_nom = df['MC trial'].nunique()

        # Line plot for each trial
        for trial in range(1,num_mc_trials_plus_nom):
            df_sp_trial = df[df["MC trial"] == trial].sort_values(by=x_col) # sorting is to make sure that the lines are drawn properly
            plt.plot(df_sp_trial[x_col], df_sp_trial[y_col], color='grey', alpha=0.1, linewidth=1, zorder = 0.9)  # 'linewidth' controls the thickness of the line, zorder puts these lines below the fill betweens!


        plt.title(f'Cumulative DVH Quantile Regression - {patientUID} - {bx_struct_roi}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        #handles, labels = plt.gca().get_legend_handles_labels()
        #plt.legend(handles[:1], labels[:1], title="Legend", frameon=True, facecolor='white')
        plt.legend(['5th-25th Percentile', '25th-75th Percentile', '75th-95th Percentile', 'Nominal'], loc='best', facecolor = 'white')
        plt.tight_layout()

        return fig

    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]

    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
        bx_struct_roi = specific_bx_structure["ROI"]
        cumulative_dvh_pandas_dataframe = specific_bx_structure["Output data frames"]["Cumulative DVH by MC trial"]

        df = cumulative_dvh_pandas_dataframe

        fig = plot_filled_quantiles(df, 'Dose (Gy)', 'Percent volume', patientUID, bx_struct_roi)
        
        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

        fig.savefig(svg_dose_fig_file_path, format='svg')

        # clean up for memory
        plt.close(fig)



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

        cumulative_dvh_pandas_dataframe = specific_bx_structure["Output data frames"]["Cumulative DVH by MC trial"]
        cumulative_dvh_pandas_dataframe_N_trials = cumulative_dvh_pandas_dataframe[cumulative_dvh_pandas_dataframe["MC trial"] < num_cumulative_dvh_plots_to_show]


        fig_global = px.line(cumulative_dvh_pandas_dataframe_N_trials, x="Dose (Gy)", y="Percent volume", color = "MC trial", width  = svg_image_width, height = svg_image_height)
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
                                                                     "R (Bx frame)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                                     "Z (Bx frame)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                     "Mean probability (binom est)": binom_est_point_wise_for_pd_data_frame_list, 
                                                                     "Total successes": total_successes_point_wise_for_pd_data_frame_list, 
                                                                     "STD err": std_err_point_wise_for_pd_data_frame_list,
                                                                     "Nominal containment": nominal_point_wise_dor_pd_data_frame_list
                                                                     }
        
        containment_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_output_dict_by_MC_trial_for_pandas_data_frame)

        specific_bx_structure["Output data frames"]["Containment ouput by bx point"] = containment_output_by_MC_trial_pandas_data_frame
        #specific_bx_structure["Output dicts for data frames"]["Containment ouput by bx point"] = containment_output_dict_by_MC_trial_for_pandas_data_frame

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
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Z (Bx frame)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Mean probability (binom est)"], 
                    eval_x = z_vals_to_evaluate, N=num_bootstraps_for_regression_plots_input, conf_interval=0.95
                )
            elif regression_type_ans == False:
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit, \
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_lower, \
                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Z (Bx frame)"], 
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
                                        x="Z (Bx frame)", 
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
                                        x="Z (Bx frame)", 
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
                xaxis_title='Z (Bx frame)',
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
                                                                    "R (Bx frame)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                                    "Z (Bx frame)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                    "Mean probability (binom est)": binom_est_point_wise_for_pd_data_frame_list, 
                                                                    "STD err": std_err_point_wise_for_pd_data_frame_list,
                                                                    "Nominal containment": nominal_point_wise_for_pd_data_frame_list,
                                                                    "CI lower vals": binom_est_lower_CI_point_wise_for_pd_data_frame_list,
                                                                    "CI upper vals": binom_est_upper_CI_point_wise_for_pd_data_frame_list
                                                                    }
        
        containment_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_output_dict_by_MC_trial_for_pandas_data_frame)

        specific_bx_structure["Output data frames"]["Mutual containment output by bx point"] = containment_output_by_MC_trial_pandas_data_frame
        #specific_bx_structure["Output dicts for data frames"]["Mutual containment output by bx point"] = containment_output_dict_by_MC_trial_for_pandas_data_frame

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
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Z (Bx frame)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Mean probability (binom est)"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95
                )
                miss_structure_probability_vs_axial_Z_NPKR_lower_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Z (Bx frame)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "CI lower vals"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95
                )
                miss_structure_probability_vs_axial_Z_NPKR_upper_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Z (Bx frame)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "CI upper vals"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95
                )
            elif regression_type_ans == False:
                miss_structure_probability_vs_axial_Z_NPKR_binom_est_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Z (Bx frame)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Mean probability (binom est)"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95, bandwidth = NPKR_bandwidth
                )
                miss_structure_probability_vs_axial_Z_NPKR_lower_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Z (Bx frame)"], 
                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "CI lower vals"], 
                    eval_x = z_vals_to_evaluate, N=1, conf_interval=0.95, bandwidth = NPKR_bandwidth
                )
                miss_structure_probability_vs_axial_Z_NPKR_upper_CI_fit, \
                bootstrap_lower, \
                bootstrap_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                    parallel_pool,
                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == roi_to_regress, "Z (Bx frame)"], 
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
                                        x="Z (Bx frame)", 
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
                                        x="Z (Bx frame)", 
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
                xaxis_title='Z (Bx frame)',
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
                                                bx_ref,
                                                dil_ref,
                                                cancer_tissue_label,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                structure_miss_probability_roi,
                                                tissue_class_sobol_global_plot_bool_dict,
                                                box_plot_points_option,
                                                notch_option,
                                                boxmean_option
                                                ):
    

    color_discrete_map_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}

    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict["Global"]
    fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA info"]["FANOVA: sobol var names by index"]
    fanova_sobol_indices_names_FO = [x + ' FO' for x in fanova_sobol_indices_names_by_index]
    fanova_sobol_indices_names_TO = [x + ' TO' for x in fanova_sobol_indices_names_by_index]

    #num_biopsies = master_structure_info_dict["Global"]["Num biopsies"]
    dataframes_list = []
    for patientUID,pydicom_item in master_structure_reference_dict.items():  
        for bx_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            sp_bx_sobol_containment_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol containment (DIL tissue) dataframe"])
            dataframes_list.append(sp_bx_sobol_containment_dataframe)

    """
    dataframes_list_other = []
    for patientUID,pydicom_item in master_structure_reference_dict.items():  
        for bx_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            sp_bx_sobol_containment_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol containment dataframe"][specific_bx_structure["FANOVA: sobol containment dataframe"]["Relative structure type"] == dil_ref])
            dataframes_list_other.append(sp_bx_sobol_containment_dataframe)
    """

    grand_sobol_dataframe = cudf.concat(dataframes_list, ignore_index=True)       
    
    if tissue_class_sobol_global_plot_bool_dict["Global FO"] == True:
        
        num_uncertainty_vars = len(fanova_sobol_indices_names_by_index)
        comb_seed_list = list(range(0,num_uncertainty_vars))
        combs_list_for_p_vals = []
        for subset in itertools.combinations(comb_seed_list, 2):
            combs_list_for_p_vals.append(list(subset))

        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        column_names_first_order_sobol_list.append("Simulated bx bool")
        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_first_order_sobol_list]
        grand_sobol_dataframe_first_order_non_sim_only = grand_sobol_dataframe_first_order_all[grand_sobol_dataframe_first_order_all["Simulated bx bool"] == False]
        grand_sobol_dataframe_first_order_sim_only = grand_sobol_dataframe_first_order_all[grand_sobol_dataframe_first_order_all["Simulated bx bool"] == True]
        
        fig_non_sim = go.Figure()
        for FO_uncertainty_type in fanova_sobol_indices_names_FO:
            fig_non_sim.add_trace(go.Box(
                y = grand_sobol_dataframe_first_order_non_sim_only[FO_uncertainty_type].to_numpy(),
                name = FO_uncertainty_type,
                marker_color = color_discrete_map_dict[False],
                boxpoints = box_plot_points_option,
                notched = notch_option,
                boxmean = boxmean_option,
            ))
        fig_non_sim = plotting_funcs.fix_plotly_grid_lines(fig_non_sim, y_axis = True, x_axis = False)
        fig_non_sim.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices (tissue classification, '+cancer_tissue_label+') (non-sim only)',
            hovermode="x unified"
        )
        fig_non_sim = add_p_value_annotation(fig_non_sim, 
                            combs_list_for_p_vals, 
                            subplot=None,
                            _format=dict(interline=0.07, text_height=1.05, color='black')
                            )
        fig_non_sim.update_layout(
            margin=dict(t=50*len(combs_list_for_p_vals))
            )

        fig_sim = go.Figure()
        for FO_uncertainty_type in fanova_sobol_indices_names_FO:
            fig_sim.add_trace(go.Box(
                y = grand_sobol_dataframe_first_order_sim_only[FO_uncertainty_type].to_numpy(),
                name = FO_uncertainty_type,
                marker_color = color_discrete_map_dict[True],
                boxpoints = box_plot_points_option,
                notched = notch_option,
                boxmean = boxmean_option,
            ))
        fig_sim = plotting_funcs.fix_plotly_grid_lines(fig_sim, y_axis = True, x_axis = False)
        fig_sim.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices (tissue classification, '+cancer_tissue_label+') (sim only)',
            hovermode="x unified"
        )
        fig_sim = add_p_value_annotation(fig_sim, 
                            combs_list_for_p_vals,
                            subplot=None, 
                            _format=dict(interline=0.07, text_height=1.05, color='black')
                            )
        fig_sim.update_layout(
            margin=dict(t=50*len(combs_list_for_p_vals))
            )

        # cant use plotly express for p value generation because only creates one trace
        """
        fig_sim = px.box(grand_sobol_dataframe_first_order_sim_only, 
                     points = box_plot_points_option,
                     color = "Simulated bx bool",
                     color_discrete_map = color_discrete_map_dict)
        fig_sim = add_p_value_annotation(fig_sim, 
                            combs_list_for_p_vals, 
                            _format=dict(interline=0.07, text_height=1.05, color='black')
                            )
        fig_non_sim = px.box(grand_sobol_dataframe_first_order_non_sim_only, 
                     points = box_plot_points_option,
                     color = "Simulated bx bool",
                     color_discrete_map = color_discrete_map_dict)
        fig_non_sim = add_p_value_annotation(fig_non_sim, 
                            combs_list_for_p_vals, 
                            _format=dict(interline=0.07, text_height=1.05, color='black')
                            )
        """
        
        
        # old code
        """
        

        column_names_first_order_sobol_list = [name+' FO' for name in fanova_sobol_indices_names_by_index]
        column_names_first_order_sobol_list.append("Simulated bx bool")
        grand_sobol_dataframe_first_order_all = grand_sobol_dataframe[column_names_first_order_sobol_list]
        grand_sobol_dataframe_first_order_non_sim_only = grand_sobol_dataframe_first_order_all[grand_sobol_dataframe_first_order_all["Simulated bx bool"] == False]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_non_sim_only, 
                     points = box_plot_points_option,
                     color = "Simulated bx bool",
                     color_discrete_map = color_discrete_map_dict)
        #fig = fig.update_traces(marker_color = 'rgba(227, 27, 35,1)') 
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices (tissue classification, '+cancer_tissue_label+') (non-sim only)',
            hovermode="x unified"
        )
        """

        general_plot_name_string = general_plot_name_string_dict["Global FO"]

        svg_dose_fig_name = general_plot_name_string+'_sim.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig_sim.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'_sim.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig_sim.write_html(html_dose_fig_file_path)

        svg_dose_fig_name = general_plot_name_string+'_nonsim.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig_non_sim.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'_nonsim.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig_non_sim.write_html(html_dose_fig_file_path)

    if tissue_class_sobol_global_plot_bool_dict["Global TO"] == True:
        general_plot_name_string = general_plot_name_string_dict["Global TO"]

        column_names_total_order_sobol_list = [name+' TO' for name in fanova_sobol_indices_names_by_index]
        column_names_total_order_sobol_list.append("Simulated bx bool")

        grand_sobol_dataframe_total_order_all = grand_sobol_dataframe[column_names_total_order_sobol_list]
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_total_order_all, 
                     points = box_plot_points_option,
                     color = "Simulated bx bool",
                     color_discrete_map = color_discrete_map_dict)
        #fig = fig.update_traces(marker_color = 'rgba(227, 27, 35,1)') 
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='Total order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of total order Sobol indices (tissue classification, ' + cancer_tissue_label + ')',
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
        
        # global sobol first order box plot
        fig = px.box(grand_sobol_dataframe_first_order_all, 
                     points = box_plot_points_option, 
                     color = "Simulated bx bool",
                     color_discrete_map=color_discrete_map_dict)
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        fig.update_layout(
            yaxis_title='First order Sobol index value (S_i)',
            xaxis_title='Index',
            title='Distribution of first order Sobol indices by biopsy class (tissue classification, ' + cancer_tissue_label + ')',
            hovermode="x unified"
        )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 



def production_plot_sobol_indices_each_biopsy_containment(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                master_structure_info_dict,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                tissue_class_sobol_per_biopsy_plot_bool_dict
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA info"]["FANOVA: sobol var names by index"]
  
    

    if tissue_class_sobol_per_biopsy_plot_bool_dict["FO"] == True:

        general_plot_name_string = general_plot_name_string_dict["FO"]

        for bx_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            bx_struct_roi = specific_bx_structure['ROI'] 
            sp_bx_sobol_containment_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol containment dataframe"])

            fanova_sobol_indices_names_FO = [x + ' FO' for x in fanova_sobol_indices_names_by_index]
            fanova_sobol_indices_names_FO_SE = [x + ' FO SE' for x in fanova_sobol_indices_names_by_index]
            id_vars_for_merge = ['Patient','Bx ROI','Simulated bx bool','Relative structure ROI','Relative structure index','Relative structure type']
            replacement_dict_FO = {x + ' FO': x for x in fanova_sobol_indices_names_by_index}
            replacement_dict_FO_SE = {x + ' FO SE': x for x in fanova_sobol_indices_names_by_index}

            melted_grand_sobol_dataframe_FO = sp_bx_sobol_containment_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_FO, var_name='FO var', value_name='FO val')
            melted_grand_sobol_dataframe_FO['FO var'] = melted_grand_sobol_dataframe_FO['FO var'].replace(replacement_dict_FO) 
            melted_grand_sobol_dataframe_FO_SE = sp_bx_sobol_containment_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_FO_SE, var_name='FO var', value_name='SE val')
            melted_grand_sobol_dataframe_FO_SE['FO var'] = melted_grand_sobol_dataframe_FO_SE['FO var'].replace(replacement_dict_FO_SE)
            merged_grand_sobol_dataframe_FO = melted_grand_sobol_dataframe_FO.merge(melted_grand_sobol_dataframe_FO_SE,how='outer')
            
            fig_sobol_per_bx = px.scatter(merged_grand_sobol_dataframe_FO, y='FO val', x='FO var', color='Relative structure ROI', error_y = 'SE val')

            fig_sobol_per_bx = plotting_funcs.fix_plotly_grid_lines(fig_sobol_per_bx, y_axis = True, x_axis = False)
            fig_sobol_per_bx.update_layout(scattermode="group", scattergap=0.75)
            fig_sobol_per_bx.update_xaxes(categoryorder='array', categoryarray= fanova_sobol_indices_names_by_index)
            fig_sobol_per_bx.update_layout(
                yaxis_title='First order Sobol index value (S_i)',
                xaxis_title='Index',
                title='First order Sobol indices (tissue classification) (' + patientUID + ', ' + bx_struct_roi + ')',
                hovermode="x unified"
            )

            svg_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.svg'
            svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
            fig_sobol_per_bx.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.html'
            html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
            fig_sobol_per_bx.write_html(html_dose_fig_file_path)

    if tissue_class_sobol_per_biopsy_plot_bool_dict["TO"] == True:

        general_plot_name_string = general_plot_name_string_dict["TO"]

        for bx_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            bx_struct_roi = specific_bx_structure['ROI'] 
            sp_bx_sobol_containment_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol containment dataframe"])

            fanova_sobol_indices_names_TO = [x + ' TO' for x in fanova_sobol_indices_names_by_index]
            fanova_sobol_indices_names_TO_SE = [x + ' TO SE' for x in fanova_sobol_indices_names_by_index]
            id_vars_for_merge = ['Patient','Bx ROI','Simulated bx bool','Relative structure ROI','Relative structure index','Relative structure type']
            replacement_dict_TO = {x + ' TO': x for x in fanova_sobol_indices_names_by_index}
            replacement_dict_TO_SE = {x + ' TO SE': x for x in fanova_sobol_indices_names_by_index}

            melted_grand_sobol_dataframe_TO = sp_bx_sobol_containment_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_TO, var_name='TO var', value_name='TO val')
            melted_grand_sobol_dataframe_TO['TO var'] = melted_grand_sobol_dataframe_TO['TO var'].replace(replacement_dict_TO) 
            melted_grand_sobol_dataframe_TO_SE = sp_bx_sobol_containment_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_TO_SE, var_name='TO var', value_name='SE val')
            melted_grand_sobol_dataframe_TO_SE['TO var'] = melted_grand_sobol_dataframe_TO_SE['TO var'].replace(replacement_dict_TO_SE)
            merged_grand_sobol_dataframe_TO = melted_grand_sobol_dataframe_TO.merge(melted_grand_sobol_dataframe_TO_SE,how='outer')
            
            fig_sobol_per_bx = px.scatter(merged_grand_sobol_dataframe_TO, y='TO val', x='TO var', color='Relative structure ROI', error_y = 'SE val')

            fig_sobol_per_bx = plotting_funcs.fix_plotly_grid_lines(fig_sobol_per_bx, y_axis = True, x_axis = False)
            fig_sobol_per_bx.update_layout(scattermode="group", scattergap=0.75)
            fig_sobol_per_bx.update_xaxes(categoryorder='array', categoryarray= fanova_sobol_indices_names_by_index)
            fig_sobol_per_bx.update_layout(
                yaxis_title='Total order Sobol index value (Stot_i)',
                xaxis_title='Index',
                title='Total order Sobol indices (tissue classification) (' + patientUID + ', ' + bx_struct_roi + ')',
                hovermode="x unified"
            )

            svg_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.svg'
            svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
            fig_sobol_per_bx.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.html'
            html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
            fig_sobol_per_bx.write_html(html_dose_fig_file_path)   



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
    fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA info"]["FANOVA: sobol var names by index"]
    
    #num_biopsies = master_structure_info_dict["Global"]["Num biopsies"]
    dataframes_list = []
    for patientUID,pydicom_item in master_structure_reference_dict.items():  
        for specific_bx_structure in pydicom_item[bx_structs]:
            
            sp_bx_sobol_dose_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol dose dataframe"])
            dataframes_list.append(sp_bx_sobol_dose_dataframe)

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





def production_plot_sobol_indices_each_biopsy_dosimetry(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                master_structure_info_dict,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                dosimetry_sobol_per_biopsy_plot_bool_dict
                                                ):
    
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
    fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA info"]["FANOVA: sobol var names by index"]

    if dosimetry_sobol_per_biopsy_plot_bool_dict["FO"] == True:

        general_plot_name_string = general_plot_name_string_dict["FO"]

        for bx_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            bx_struct_roi = specific_bx_structure['ROI'] 
            sp_bx_sobol_dose_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol dose dataframe"])

            fanova_sobol_indices_names_FO = [x + ' FO' for x in fanova_sobol_indices_names_by_index]
            fanova_sobol_indices_names_FO_SE = [x + ' FO SE' for x in fanova_sobol_indices_names_by_index]
            id_vars_for_merge = ['Patient','Bx ROI','Simulated bx bool','Function output key']
            replacement_dict_FO = {x + ' FO': x for x in fanova_sobol_indices_names_by_index}
            replacement_dict_FO_SE = {x + ' FO SE': x for x in fanova_sobol_indices_names_by_index}

            melted_grand_sobol_dataframe_FO = sp_bx_sobol_dose_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_FO, var_name='FO var', value_name='FO val')
            melted_grand_sobol_dataframe_FO['FO var'] = melted_grand_sobol_dataframe_FO['FO var'].replace(replacement_dict_FO) 
            melted_grand_sobol_dataframe_FO_SE = sp_bx_sobol_dose_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_FO_SE, var_name='FO var', value_name='SE val')
            melted_grand_sobol_dataframe_FO_SE['FO var'] = melted_grand_sobol_dataframe_FO_SE['FO var'].replace(replacement_dict_FO_SE)
            merged_grand_sobol_dataframe_FO = melted_grand_sobol_dataframe_FO.merge(melted_grand_sobol_dataframe_FO_SE,how='outer')
            
            fig_sobol_per_bx = px.scatter(merged_grand_sobol_dataframe_FO, y='FO val', x='FO var', color='Function output key', error_y = 'SE val')

            fig_sobol_per_bx = plotting_funcs.fix_plotly_grid_lines(fig_sobol_per_bx, y_axis = True, x_axis = False)
            fig_sobol_per_bx.update_layout(scattermode="group", scattergap=0.75)
            fig_sobol_per_bx.update_xaxes(categoryorder='array', categoryarray= fanova_sobol_indices_names_by_index)
            fig_sobol_per_bx.update_layout(
                yaxis_title='First order Sobol index value (S_i)',
                xaxis_title='Index',
                title='First order Sobol indices (dosimetry) (' + patientUID + ', ' + bx_struct_roi + ')',
                hovermode="x unified"
            )

            svg_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.svg'
            svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
            fig_sobol_per_bx.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.html'
            html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
            fig_sobol_per_bx.write_html(html_dose_fig_file_path)

    if dosimetry_sobol_per_biopsy_plot_bool_dict["TO"] == True:

        general_plot_name_string = general_plot_name_string_dict["TO"]

        for bx_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            bx_struct_roi = specific_bx_structure['ROI'] 
            sp_bx_sobol_dose_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sobol dose dataframe"])

            fanova_sobol_indices_names_TO = [x + ' TO' for x in fanova_sobol_indices_names_by_index]
            fanova_sobol_indices_names_TO_SE = [x + ' TO SE' for x in fanova_sobol_indices_names_by_index]
            id_vars_for_merge = ['Patient','Bx ROI','Simulated bx bool','Function output key']
            replacement_dict_TO = {x + ' TO': x for x in fanova_sobol_indices_names_by_index}
            replacement_dict_TO_SE = {x + ' TO SE': x for x in fanova_sobol_indices_names_by_index}

            melted_grand_sobol_dataframe_TO = sp_bx_sobol_dose_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_TO, var_name='TO var', value_name='TO val')
            melted_grand_sobol_dataframe_TO['TO var'] = melted_grand_sobol_dataframe_TO['TO var'].replace(replacement_dict_TO) 
            melted_grand_sobol_dataframe_TO_SE = sp_bx_sobol_dose_dataframe.melt(id_vars = id_vars_for_merge, value_vars = fanova_sobol_indices_names_TO_SE, var_name='TO var', value_name='SE val')
            melted_grand_sobol_dataframe_TO_SE['TO var'] = melted_grand_sobol_dataframe_TO_SE['TO var'].replace(replacement_dict_TO_SE)
            merged_grand_sobol_dataframe_TO = melted_grand_sobol_dataframe_TO.merge(melted_grand_sobol_dataframe_TO_SE,how='outer')
            
            fig_sobol_per_bx = px.scatter(merged_grand_sobol_dataframe_TO, y='TO val', x='TO var', color='Function output key', error_y = 'SE val')

            fig_sobol_per_bx = plotting_funcs.fix_plotly_grid_lines(fig_sobol_per_bx, y_axis = True, x_axis = False)
            fig_sobol_per_bx.update_layout(scattermode="group", scattergap=0.75)
            fig_sobol_per_bx.update_xaxes(categoryorder='array', categoryarray= fanova_sobol_indices_names_by_index)
            fig_sobol_per_bx.update_layout(
                yaxis_title='Total order Sobol index value (Stot_i)',
                xaxis_title='Index',
                title='Total order Sobol indices (dosimetry) (' + patientUID + ', ' + bx_struct_roi + ')',
                hovermode="x unified"
            )

            svg_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.svg'
            svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
            fig_sobol_per_bx.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_dose_fig_name = bx_struct_roi+' - '+general_plot_name_string+'.html'
            html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
            fig_sobol_per_bx.write_html(html_dose_fig_file_path)



def production_plot_tissue_patient_cohort(patient_cohort_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    general_plot_name_string,
                                    cohort_output_figures_dir,
                                    box_plot_points_option = 'outliers',
                                    notch_option = True,
                                    boxmean_option = 'sd'
                                    ):
    
    color_discrete_map_sim_or_no_sim_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}

    tissue_types_list = patient_cohort_dataframe["Tissue type"].unique()

    fig = make_subplots(rows=1, 
                        cols=len(tissue_types_list)
                        )
    
     
    for index in range(len(tissue_types_list)):
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == True)]["Mean probability"],
            name = tissue_types_list[index] + ' simulated',
            marker_color = color_discrete_map_sim_or_no_sim_dict[True],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option,
            #customdata = np.full(patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == True)]["Mean probability"].size, np.std(patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == True)]["Mean probability"])) 
        ), row =1 , col = index+1)
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == False)]["Mean probability"],
            name = tissue_types_list[index] + ' actual',
            marker_color = color_discrete_map_sim_or_no_sim_dict[False],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option,
            #customdata = np.full(patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == False)]["Mean probability"].size, np.std(patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == False)]["Mean probability"]))
        ), row =1 , col = index+1)

    """
    fig.append_trace(go.Box(
        y = patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[1]) & (patient_cohort_dataframe["Simulated bool"] == True)]["Mean probability"],
        name = tissue_types_list[1] + ' simulated',
        marker_color = color_discrete_map_sim_or_no_sim_dict[True],
        boxpoints = box_plot_points_option,
        notched = notch_option,
        boxmean = boxmean_option
    ), row =1 , col =2)
    fig.append_trace(go.Box(
        y = patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[1]) & (patient_cohort_dataframe["Simulated bool"] == False)]["Mean probability"],
        name = tissue_types_list[1] + ' actual',  
        marker_color = color_discrete_map_sim_or_no_sim_dict[False],
        boxpoints = box_plot_points_option,
        notched = notch_option,
        boxmean = boxmean_option
    ), row =1 , col =2)
    """

    """
    std_dev = {}
    for trace in fig.data:
        name = trace['name']
        data = trace['y']
        std = np.std(data)
        std_dev[name] = std

    fig.update_traces(hovertemplate='SD: %{customdata}')
    """
    
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    for index in range(len(tissue_types_list)):
        fig.update_xaxes(title_text="Tissue classification", row=1, col=index+1)
        fig.update_yaxes(title_text="Probability", range = [0,1.01], row=1, col=index+1)
    
    fig.update_layout(
        #yaxis_title='Probability',
        #xaxis_title='Tissue classification',
        title_text='Patient cohort tissue classification probability (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        #hovermode="x unified"
    )
    #fig.update_layout(
    #boxmode='group' # group together boxes of the different traces for each value of x
    #)
    #fig.update_traces(sd = np.std())
    #fig.update_traces(hovertemplate='SD: %{customdata}')

    for index in range(len(tissue_types_list)):
        fig = add_p_value_annotation(fig, 
                            [[0,1]], 
                            subplot=index +1, 
                            _format=dict(interline=0.07, text_height=1.05, color='black')
                            )
    """
    fig = add_p_value_annotation(fig, 
                           [[0,1]], 
                           subplot=2, 
                           _format=dict(interline=0.07, text_height=1.07, color='black')
                           )
    """

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 



def production_plot_tissue_patient_cohort_NEW(master_cohort_patient_data_and_dataframes,
                                              master_st_ds_info_dict,
                                              miss_structure_complement_label,
                                              all_ref_key,
                                              color_discrete_map_by_sim_type,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string,
                                            cohort_output_figures_dir,
                                            box_plot_points_option = 'outliers',
                                            notch_option = True,
                                            boxmean_option = 'sd'
                                            ):
    
    num_biopsies_by_bx_type_dict = master_st_ds_info_dict["Global"]["Num biopsies by bx type dict"]

    all_patients_global_containment_scores_by_tissue_class = copy.deepcopy(master_cohort_patient_data_and_dataframes['Dataframes']['Cohort: tissue class global scores (tissue type)'])


    tissue_types_list = all_patients_global_containment_scores_by_tissue_class['Structure ROI'].unique().tolist()
    tissue_types_list.remove(miss_structure_complement_label)

    simulated_types_list = all_patients_global_containment_scores_by_tissue_class["Simulated type"].unique()


    metric_types_list = ['Global mean binom est', 'Global min binom est', 'Global max binom est']

    fig = make_subplots(rows=len(tissue_types_list), 
                        cols=len(metric_types_list)
                        )

    for metric_type_index, metric_type in enumerate(metric_types_list):

        for tissue_type_index, tissue_type in enumerate(tissue_types_list):
            for sim_type_index, sim_type in enumerate(simulated_types_list):
                fig.append_trace(go.Box(
                    y = all_patients_global_containment_scores_by_tissue_class[(all_patients_global_containment_scores_by_tissue_class['Structure ROI'] == tissue_type) & (all_patients_global_containment_scores_by_tissue_class["Simulated type"] == sim_type)][metric_type],
                    name = tissue_type +' - '+  sim_type,
                    marker_color = color_discrete_map_by_sim_type[sim_type],
                    boxpoints = box_plot_points_option,
                    notched = notch_option,
                    boxmean = boxmean_option,
                    #customdata = np.full(patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == True)]["Mean probability"].size, np.std(patient_cohort_dataframe[(patient_cohort_dataframe["Tissue type"] == tissue_types_list[index]) & (patient_cohort_dataframe["Simulated bool"] == True)]["Mean probability"])) 
                ), row =tissue_type_index + 1 , col = metric_type_index + 1)

    
        
        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
        for index in range(len(tissue_types_list)):
            fig.update_xaxes(title_text="Tissue classification", row=1, col=index+1)
            fig.update_yaxes(title_text="Probability", range = [0,1.01], row=1, col=index+1)
        

        text_1 = 'Patient cohort tissue classification probability'
        num_biopsies_by_type_string_list = [bpsy_type +': '+ str(num_bxs_sp_type) for bpsy_type, num_bxs_sp_type in num_biopsies_by_bx_type_dict.items()]
        text_2 = ', '.join(num_biopsies_by_type_string_list)

        text_list_for_annotation = [text_1,text_2]
        fig_description_text_for_annotation = ' | '.join(text_list_for_annotation)
        fig.add_annotation(text=fig_description_text_for_annotation,
                                xref="paper", 
                                yref="paper",
                                x=0.99, 
                                y=1.4, 
                                showarrow=False,
                                font=dict(family="Courier New, monospace", 
                                            size=16, 
                                            color="#000000"),
                                bordercolor="#000000",
                                borderwidth=2,
                                borderpad=4,
                                bgcolor="#ffffff",
                                opacity=1
                                )


        num_bpsy_types = len(simulated_types_list)
        comb_seed_list = list(range(0,num_bpsy_types))
        combs_list_for_p_vals = []
        for subset in itertools.combinations(comb_seed_list, 2):
            combs_list_for_p_vals.append(list(subset))



        for tissue_type_index, tissue_type in enumerate(tissue_types_list):
            fig = add_p_value_annotation(fig, 
                                combs_list_for_p_vals, 
                                subplot=tissue_type_index +1, 
                                _format=dict(interline=0.07, text_height=1.05, color='black')
                                )
            
        fig.update_layout(
            margin=dict(t=60*len(combs_list_for_p_vals))
            )

        svg_dose_fig_name = general_plot_name_string+'.svg'
        svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+'.html'
        html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 
        #print('test')



def production_plot_tissue_patient_cohort_NEW_v2(master_cohort_patient_data_and_dataframes,
                                              master_st_ds_info_dict,
                                              miss_structure_complement_label,
                                              all_ref_key,
                                              color_discrete_map_by_sim_type,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string,
                                            cohort_output_figures_dir,
                                            box_plot_points_option = 'outliers',
                                            notch_option = True,
                                            boxmean_option = 'sd'
                                            ):

    num_biopsies_by_bx_type_dict = master_st_ds_info_dict["Global"]["Num biopsies by bx type dict"]

    all_patients_global_containment_scores_by_tissue_class = copy.deepcopy(master_cohort_patient_data_and_dataframes['Dataframes']['Cohort: tissue class global scores (tissue type)'])

    tissue_types_list = all_patients_global_containment_scores_by_tissue_class['Structure ROI'].unique().tolist()
    tissue_types_list.remove(miss_structure_complement_label)

    simulated_types_list = all_patients_global_containment_scores_by_tissue_class["Simulated type"].unique()

    metric_types_list = ['Global mean binom est', 'Global min binom est', 'Global max binom est']

    # Create subplots with increased spacing
    fig = make_subplots(
        rows=len(tissue_types_list), 
        cols=len(metric_types_list),
        vertical_spacing=0.2,  # Increase vertical spacing
        horizontal_spacing=0.05  # Increase horizontal spacing
    )
    for metric_type_index, metric_type in enumerate(metric_types_list):
        for tissue_type_index, tissue_type in enumerate(tissue_types_list):
            for sim_type_index, sim_type in enumerate(simulated_types_list):
                fig.append_trace(go.Box(
                    y=all_patients_global_containment_scores_by_tissue_class[
                        (all_patients_global_containment_scores_by_tissue_class['Structure ROI'] == tissue_type) & 
                        (all_patients_global_containment_scores_by_tissue_class["Simulated type"] == sim_type)][metric_type],
                    name= sim_type,
                    marker_color=color_discrete_map_by_sim_type[sim_type],
                    boxpoints=box_plot_points_option,
                    notched=notch_option,
                    boxmean=boxmean_option,
                ), row=tissue_type_index + 1, col=metric_type_index + 1)

    # Fix plotly grid lines
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis=True, x_axis=False)

    # Update axis titles
    for metric_type_index, metric_type in enumerate(metric_types_list):
        for tissue_type_index, tissue_type in enumerate(tissue_types_list):
            fig.update_xaxes(title_text="Biopsy group", row=tissue_type_index + 1, col=metric_type_index+1)
            fig.update_yaxes(title_text="Probability", range=[0, 1.01], row=tissue_type_index + 1, col=metric_type_index + 1)
            # Access the domain of the specific subplot
            x_domain = fig['layout'][f'xaxis{metric_type_index + 1 + tissue_type_index * len(metric_types_list)}']['domain']
            y_domain = fig['layout'][f'yaxis{metric_type_index + 1 + tissue_type_index * len(metric_types_list)}']['domain']
            
            fig.add_annotation(
                x=x_domain[0] + (x_domain[1] - x_domain[0]) / 2, 
                y=y_domain[1] + 0.11,
                xref="paper",
                yref="paper",
                text=f"{tissue_type} - {metric_type}",
                showarrow=False,
                font=dict(size=14),
                xanchor='center'
            )


    text_1 = 'Patient cohort tissue classification probability'
    num_biopsies_by_type_string_list = [bpsy_type + ': ' + str(num_bxs_sp_type) for bpsy_type, num_bxs_sp_type in num_biopsies_by_bx_type_dict.items()]
    text_2 = ', '.join(num_biopsies_by_type_string_list)

    text_list_for_annotation = [text_1, text_2]
    fig_description_text_for_annotation = ' | '.join(text_list_for_annotation)
    fig.add_annotation(text=fig_description_text_for_annotation,
                    xref="paper",
                    yref="paper",
                    x=0.99,
                    y=-0.1,
                    showarrow=False,
                    font=dict(family="Courier New, monospace",
                                size=16,
                                color="#000000"),
                    bordercolor="#000000",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ffffff",
                    opacity=1
                    )

    num_bpsy_types = len(simulated_types_list)
    comb_seed_list = list(range(0, num_bpsy_types))
    combs_list_for_p_vals = []
    for subset in itertools.combinations(comb_seed_list, 2):
        combs_list_for_p_vals.append(list(subset))

    subplot_index = 1
    for metric_type_index, metric_type in enumerate(metric_types_list):
        for tissue_type_index, tissue_type in enumerate(tissue_types_list):
            fig = add_p_value_annotation_v2(fig,
                                        combs_list_for_p_vals,
                                        subplot=subplot_index,
                                        _format=dict(color='black')
                                        )
            subplot_index += 1

    fig = add_significance_annotation(fig, x=0, y=-0.1, _format=dict(color='black'))

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 


def production_plot_tissue_volume_box_plots_patient_cohort_NEW(master_cohort_patient_data_and_dataframes,
                                              master_st_ds_info_dict,
                                              color_discrete_map_by_sim_type,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string,
                                            cohort_output_figures_dir,
                                            box_plot_points_option = 'outliers',
                                            notch_option = True,
                                            boxmean_option = 'sd'
                                            ):
    
    num_biopsies_by_bx_type_dict = master_st_ds_info_dict["Global"]["Num biopsies by bx type dict"]

    cohort_tissue_volume_above_threshold_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: tissue volume above threshold"] 


    probability_threshold_list = cohort_tissue_volume_above_threshold_dataframe['Probability threshold'].unique().tolist()

    simulated_types_list = cohort_tissue_volume_above_threshold_dataframe["Bx type"].unique()
    
    fig = make_subplots(rows=1, 
                        cols=len(probability_threshold_list)
                        )

    for probability_threshold_index, probability_threshold in enumerate(probability_threshold_list):
        for sim_type_index, sim_type in enumerate(simulated_types_list):
            fig.append_trace(go.Box(
                y = cohort_tissue_volume_above_threshold_dataframe[(cohort_tissue_volume_above_threshold_dataframe["Probability threshold"] == probability_threshold) & 
                                                                   (cohort_tissue_volume_above_threshold_dataframe["Bx type"] == sim_type)]["Volume of DIL tissue"],
                name = f"{probability_threshold:.2f}" + ' - ' +  sim_type,
                marker_color = color_discrete_map_by_sim_type[sim_type],
                boxpoints = box_plot_points_option,
                notched = notch_option,
                boxmean = boxmean_option
            ), row =1 , col = probability_threshold_index+1)
        

    


    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    for index in range(len(probability_threshold_list)):
        fig.update_xaxes(title_text="Probability threshold", row=1, col=index+1)
        fig.update_yaxes(title_text="Tissue volume (cmm)", row=1, col=index+1)
    
    
    
    text_1 = 'Patient cohort tissue volume above given threshold'
    num_biopsies_by_type_string_list = [bpsy_type +': '+ str(num_bxs_sp_type) for bpsy_type, num_bxs_sp_type in num_biopsies_by_bx_type_dict.items()]
    text_2 = ', '.join(num_biopsies_by_type_string_list)

    text_list_for_annotation = [text_1,text_2]
    fig_description_text_for_annotation = ' | '.join(text_list_for_annotation)
    fig.add_annotation(text=fig_description_text_for_annotation,
                            xref="paper", 
                            yref="paper",
                            x=0.99, 
                            y=1.1, 
                            showarrow=False,
                            font=dict(family="Courier New, monospace", 
                                        size=16, 
                                        color="#000000"),
                            bordercolor="#000000",
                            borderwidth=2,
                            borderpad=4,
                            bgcolor="#ffffff",
                            opacity=1
                            )
    
    
    num_bpsy_types = len(simulated_types_list)
    comb_seed_list = list(range(0,num_bpsy_types))
    combs_list_for_p_vals = []
    for subset in itertools.combinations(comb_seed_list, 2):
        combs_list_for_p_vals.append(list(subset))



    for probability_threshold_index, probability_threshold in enumerate(probability_threshold_list):
        fig = add_p_value_annotation_v2(fig, 
                            combs_list_for_p_vals, 
                            subplot=probability_threshold_index +1, 
                            _format=dict(color='black')
                            )
        
    fig.update_layout(
        margin=dict(t=60*len(combs_list_for_p_vals))
        )


    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 













def production_plot_tissue_length_box_plots_patient_cohort(patient_cohort_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    general_plot_name_string,
                                    cohort_output_figures_dir,
                                    box_plot_points_option = 'outliers',
                                    notch_option = True,
                                    boxmean_option = 'sd'
                                    ):
    
    color_discrete_map_sim_or_no_sim_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}
    probability_threshold_list = patient_cohort_dataframe["Probability threshold"].unique()
    
    fig = make_subplots(rows=1, 
                        cols=len(probability_threshold_list)
                        )

    for index in range(len(probability_threshold_list)):
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[(patient_cohort_dataframe["Probability threshold"] == probability_threshold_list[index]) & (patient_cohort_dataframe["Simulated bool"] == True)]["Length estimate mean"],
            name = str(probability_threshold_list[index]) + ' simulated',
            marker_color = color_discrete_map_sim_or_no_sim_dict[True],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option
        ), row =1 , col = index+1)
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[(patient_cohort_dataframe["Probability threshold"] == probability_threshold_list[index]) & (patient_cohort_dataframe["Simulated bool"] == False)]["Length estimate mean"],
            name = str(probability_threshold_list[index]) + ' actual',
            marker_color = color_discrete_map_sim_or_no_sim_dict[False],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option
        ), row =1 , col = index+1)

    """
    fig = px.box(patient_cohort_dataframe, 
                 x="Probability threshold", 
                 y="Length estimate mean", 
                 points = box_plot_points_option, 
                 color = "Simulated bool",
                 color_discrete_map = color_discrete_map_sim_or_no_sim_dict)
    
    fig.update_traces(boxmean = boxmean_option)   
    
    
    
    fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = probability_threshold_list,
    )
    )
    
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Tissue length (mm)',
        xaxis_title='Probability threshold',
        title='Patient cohort tissue length above given threshold (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )

    fig.update_layout(
    boxmode='group' # group together boxes of the different traces for each value of x
    )
    """

    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    for index in range(len(probability_threshold_list)):
        fig.update_xaxes(title_text="Probability threshold", row=1, col=index+1)
        fig.update_yaxes(title_text="Tissue length (mm)", row=1, col=index+1)
    
    fig.update_layout(
        #yaxis_title='Probability',
        #xaxis_title='Tissue classification',
        title_text='Patient cohort tissue length above given threshold (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )
    #fig.update_layout(
    #boxmode='group' # group together boxes of the different traces for each value of x
    #)

    for index in range(len(probability_threshold_list)):
        fig = add_p_value_annotation(fig, 
                            [[0,1]], 
                            subplot=index +1, 
                            _format=dict(interline=0.07, text_height=1.05, color='black')
                            )

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 


def production_plot_tissue_length_distribution_patient_cohort(patient_cohort_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    general_plot_name_string,
                                    cohort_output_figures_dir,
                                    threshold,
                                    cdf_sp_threshold_dict,
                                    show_hist_option = False,
                                    show_rug_option = True
                                    ):
    
    x_sim = patient_cohort_dataframe[(patient_cohort_dataframe["Simulated bool"] == True) & (patient_cohort_dataframe["Probability threshold"] == threshold)]["Length estimate mean"]
    x_actual = patient_cohort_dataframe[(patient_cohort_dataframe["Simulated bool"] == False) & (patient_cohort_dataframe["Probability threshold"] == threshold)]["Length estimate mean"]

    std = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == False]["Length estimate mean"].std()

    group_labels = ['Simulated', 'Actual']

    colors = ['rgba(0, 92, 171, 1)', 'rgba(227, 27, 35,1)']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot([x_sim, x_actual], group_labels, bin_size=3.49*std*num_actual_biopsies**(-1/3),
                            curve_type='normal', # override default 'kde'
                            colors=colors,
                            show_hist = show_hist_option,  
                            show_rug = show_rug_option)
    
    # fit parameters of the normal fit, I couldnt find out how to return them directly from plotly
    mu_sim, std_sim = norm.fit(x_sim)
    mu_actual, std_actual = norm.fit(x_actual)

    fit_parameters_sim_dict = {'mu': mu_sim, 'sigma': std_sim}
    fit_parameters_actual_dict = {'mu': mu_actual, 'sigma': std_actual}


    # code for cumulative 
    cdf_sim_dict = cdf_sp_threshold_dict["CDF sim"]
    cdf_actual_dict = cdf_sp_threshold_dict["CDF actual"]

    cdf_sim_data = cdf_sim_dict["CDF"]
    cdf_sim_edges = cdf_sim_dict["Bin edges"]

    cdf_actual_data = cdf_actual_dict["CDF"]
    cdf_actual_edges = cdf_actual_dict["Bin edges"]

    fig.add_trace(go.Scatter(x=cdf_sim_edges, y=cdf_sim_data, name='CDF sim'))
    fig.add_trace(go.Scatter(x=cdf_actual_edges, y=cdf_actual_data, name='CDF actual'))



    # Add title
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Probability',
        xaxis_title='Tissue length (mm)',
        title='Patient cohort tissue length distribution P>=' + str(threshold) + ' (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )
    
    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 

    return fit_parameters_sim_dict, fit_parameters_actual_dict






def production_plot_dose_patient_cohort(patient_cohort_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    general_plot_name_string,
                                    cohort_output_figures_dir,
                                    box_plot_points_option = 'outliers',
                                    notch_option = True,
                                    boxmean_option = 'sd'
                                    ):
    
    color_discrete_map_nominal_or_global_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}

    fig = make_subplots(rows=1, 
                        cols=4
                        )
    
    nom_global_list = ['Nominal','Global']
    travelling_index = 1
    for index,type_name in enumerate(nom_global_list):
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == True][type_name+" mean dose"],
            name = type_name + ' simulated',
            marker_color = color_discrete_map_nominal_or_global_dict[True],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option
        ), row =1 , col = index+1)
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == False][type_name+" mean dose"],
            name = type_name + ' actual',
            marker_color = color_discrete_map_nominal_or_global_dict[False],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option
        ), row =1 , col = index+1)
        travelling_index = travelling_index + 1

    for index,sim_bool in enumerate([True, False]):
        if sim_bool == True:
            sim_string = ' simulated'
        else:
            sim_string = ' actual'
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == sim_bool]["Global mean dose"],
            name = 'Global' + sim_string,
            marker_color = color_discrete_map_nominal_or_global_dict[sim_bool],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option
        ), row =1 , col = travelling_index+index)
        fig.append_trace(go.Box(
            y = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == sim_bool]["Nominal mean dose"],
            name = 'Nominal' + sim_string,
            marker_color = color_discrete_map_nominal_or_global_dict[sim_bool],
            boxpoints = box_plot_points_option,
            notched = notch_option,
            boxmean = boxmean_option
        ), row =1 , col = travelling_index+index)
    
    
    

    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    for i in range(1,5):
        fig.update_xaxes(title_text="Type", row=1, col=i)
        fig.update_yaxes(title_text="Dose (Gy)", row=1, col=i)
        
    
    fig.update_layout(
        #yaxis_title='Probability',
        #xaxis_title='Tissue classification',
        title_text='Patient cohort tissue length above given threshold (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )
    #fig.update_layout(
    #boxmode='group' # group together boxes of the different traces for each value of x
    #)

    for i in range(1,5):
        fig = add_p_value_annotation(fig, 
                        [[0,1]], 
                        subplot=i, 
                        _format=dict(interline=0.07, text_height=1.05, color='black')
                        )
    

    """
    fig = go.Figure()

    fig.add_trace(go.Box(
        y = patient_cohort_dataframe["Nominal mean dose"],
        x = patient_cohort_dataframe["Simulated bool"],
        name = 'Nominal',
        marker_color = color_discrete_map_nominal_or_global_dict[False],
        boxpoints = box_plot_points_option,
        notched = notch_option,
        boxmean = boxmean_option
    ))
    fig.add_trace(go.Box(
        y = patient_cohort_dataframe["Global mean dose"],
        x = patient_cohort_dataframe["Simulated bool"],
        name = 'Global',
        marker_color = color_discrete_map_nominal_or_global_dict[True],
        boxpoints = box_plot_points_option,
        notched = notch_option,
        boxmean = boxmean_option
    ))

        
    
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Dose (Gy)',
        xaxis_title='Simulated',
        title='Patient cohort dose (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )
    fig.update_layout(
    boxmode='group' # group together boxes of the different traces for each value of x
    )
    """

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 



def production_plot_dose_distribution_patient_cohort(patient_cohort_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    general_plot_name_string,
                                    cohort_output_figures_dir,
                                    show_hist_option = False,
                                    show_rug_option = True
                                    ):
    
    x_sim = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == True]["Global mean dose"]
    x_actual = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == False]["Global mean dose"]

    std = patient_cohort_dataframe[patient_cohort_dataframe["Simulated bool"] == False]["Global mean dose"].std()

    group_labels = ['Simulated', 'Actual']

    colors = ['rgba(0, 92, 171, 1)', 'rgba(227, 27, 35,1)']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot([x_sim, x_actual], group_labels, bin_size=3.49*std*num_actual_biopsies**(-1/3),
                            curve_type='normal', # override default 'kde'
                            colors=colors,
                            show_hist = show_hist_option,  
                            show_rug = show_rug_option)
    
    # fit parameters of the normal fit, I couldnt find out how to return them directly from plotly
    mu_sim, std_sim = norm.fit(x_sim)
    mu_actual, std_actual = norm.fit(x_actual)

    fit_parameters_sim_dict = {'mu': mu_sim, 'sigma': std_sim}
    fit_parameters_actual_dict = {'mu': mu_actual, 'sigma': std_actual}

    # Add title
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Probability',
        xaxis_title='Dose (Gy)',
        title='Patient cohort global mean distribution (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )
    
    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path) 

    return fit_parameters_sim_dict, fit_parameters_actual_dict
        


def production_plot_dose_nominal_global_difference_box_patient_cohort(patient_cohort_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    general_plot_name_string,
                                    cohort_output_figures_dir,
                                    box_plot_points_option = 'outliers',
                                    notch_option = True,
                                    boxmean_option = 'sd'
                                    ):
    
    color_discrete_map_nominal_or_global_dict = {True: 'rgba(0, 92, 171, 1)', False: 'rgba(227, 27, 35,1)'}

    patient_cohort_dataframe['Nominal - global'] = patient_cohort_dataframe.apply(lambda x: x["Nominal mean dose"] - x["Global mean dose"], axis=1)
    

    fig = px.box(patient_cohort_dataframe, x="Simulated bool", y='Nominal - global', points = box_plot_points_option)
    fig.update_traces(boxmean = boxmean_option)   
    
    fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
    fig.update_layout(
        yaxis_title='Dose (Gy)',
        xaxis_title='Simulated',
        title='Patient cohort nominal mean vs global mean dose difference (N_sim_bx = '+str(num_sim_biopsies) +')'+ '(N_actual_bx = '+str(num_actual_biopsies) +')',
        hovermode="x unified"
    )
    

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

    html_dose_fig_name = general_plot_name_string+'.html'
    html_dose_fig_file_path = cohort_output_figures_dir.joinpath(html_dose_fig_name)
    fig.write_html(html_dose_fig_file_path)
        



def production_plot_guidance_maps_cumulative_projection(patientUID,
                                                        patient_sp_output_figures_dir,
                                                        pydicom_item,
                                                        all_ref_key,
                                                        svg_image_scale,
                                                        svg_image_width,
                                                        svg_image_height,
                                                        general_plot_name_string
                                                        ):


    cumulative_projection_optimization_scores_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - Cumulative projection (all points within prostate) dataframe"]
    sp_patient_centroid_optimal_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - DIL centroids optimal targeting dataframe"]
    sp_patient_optimal_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - Optimal DIL targeting dataframe"]



    list_of_cumulative_planes = cumulative_projection_optimization_scores_dataframe['Patient plane'].unique().tolist()
    for plane in list_of_cumulative_planes:
        plane_specific_cumulative_projection_optimization_scores_dataframe = cumulative_projection_optimization_scores_dataframe[cumulative_projection_optimization_scores_dataframe['Patient plane'] == plane]
        hor_axis_column_name = plane_specific_cumulative_projection_optimization_scores_dataframe.sample().reset_index(drop=True).at[0,'Coord 1 name']
        vert_axis_column_name = plane_specific_cumulative_projection_optimization_scores_dataframe.sample().reset_index(drop=True).at[0,'Coord 2 name']
        
        
        fig = go.Figure()

        # Add optimal sampling location for each DIL
        for index, row in sp_patient_optimal_dataframe.iterrows():
            fig.add_scatter(x=[row[hor_axis_column_name]],
                    y=[row[vert_axis_column_name]],
                    marker=dict(
                        color='orange',
                        size=10,
                        symbol = 'circle'
                    ),
                    text=[row["Relative DIL ID"]],
                    mode = "markers+text",
                    name=row["Relative DIL ID"]+' optimal',
                    textposition="bottom center",
                    textfont=dict(
                        family="sans serif",
                        size=12,
                        color="white"
                    )
                )

        # Add centroids of each DIL
        for index, row in sp_patient_centroid_optimal_dataframe.iterrows():
            fig.add_scatter(x=[row[hor_axis_column_name]],
                    y=[row[vert_axis_column_name]],
                    marker=dict(
                        color='yellow',
                        size=10,
                        symbol = 'circle'
                    ),
                    text=[row["Relative DIL ID"]],
                    mode = "markers+text",
                    name=row["Relative DIL ID"]+' centroid',
                    textposition="bottom center",
                    textfont=dict(
                        family="sans serif",
                        size=12,
                        color="white"
                    )
                )
            
        # Add prostate centroid
        fig.add_scatter(x=[0],
                    y=[0],
                    marker=dict(
                        color='black',
                        size=12,
                        symbol = 'circle'
                    ),
                mode='markers',
                name='Prostate centroid')  

        # Add contour plot of the values of the cumulative projection of the optimization
        fig.add_trace(
            go.Contour(
                z=plane_specific_cumulative_projection_optimization_scores_dataframe['Proportion of normal dist points contained'],
                x=plane_specific_cumulative_projection_optimization_scores_dataframe["Coordinate 1"],
                y=plane_specific_cumulative_projection_optimization_scores_dataframe["Coordinate 2"],
                colorscale=[[0, 'rgb(0,0,255)'], [0.9, 'rgb(255,0,0)'],[1, 'rgb(0,255,0)']],
                zmax = 1,
                zmin = 0,
                autocontour = False,
                contours = go.contour.Contours(type = 'levels', showlines = True, coloring = 'heatmap', showlabels = True, size = 0.1),
                connectgaps = False, 
                colorbar = go.contour.ColorBar(len = 0.5)
            ))

        # Add axes names and other annotations on the figure
        x_axis_name = hor_axis_column_name[-2]
        y_axis_name = vert_axis_column_name[-2]
        patient_pos_dict = {'X': ' (L/R)', "Y":' (A/P)', "Z": '(S/I)'}
        fig['layout']['xaxis'].update(title=x_axis_name+patient_pos_dict[x_axis_name])
        fig['layout']['yaxis'].update(title=y_axis_name+patient_pos_dict[y_axis_name])
        
        patient_plane_dict = {'XY': ' Transverse (XY)', "YZ": ' Sagittal (YZ)', "XZ": ' Coronal (XZ)',
                                'YX': ' Transverse (YX)', "ZY": ' Sagittal (ZY)', "ZX": ' Coronal (ZX)'}
        patient_plane_determiner_str = x_axis_name+y_axis_name
        

        # Add annotation 
        text_1 = "Patient: "+ patientUID
        text_2 = "Cumulative, "+patient_plane_dict[patient_plane_determiner_str]+' plane'
        text_3 = 'Prostate centroid origin (mm)'
        text_list_for_annotation = [text_1,text_2,text_3]
        fig_description_text_for_annotation = ' | '.join(text_list_for_annotation)
        fig.add_annotation(text=fig_description_text_for_annotation,
                                xref="paper", 
                                yref="paper",
                                x=0.99, 
                                y=1.1, 
                                showarrow=False,
                                font=dict(family="Courier New, monospace", 
                                            size=16, 
                                            color="#000000"),
                                bordercolor="#000000",
                                borderwidth=2,
                                borderpad=4,
                                bgcolor="#ffffff",
                                opacity=1
                                )

        
        # add annotations for the optimal positions
        fig = box_annotator_func_for_contour_plots(fig, sp_patient_optimal_dataframe, 0.99, 1.06, 'Optimal: ')
        
        # add annotations for the centroids
        fig = box_annotator_func_for_contour_plots(fig, sp_patient_centroid_optimal_dataframe,  0.99, 1.02, 'DIL centroids: ')
        

        svg_dose_fig_name = general_plot_name_string+' - (cumulative - ' +patient_plane_dict[patient_plane_determiner_str]+').svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = general_plot_name_string+' - (cumulative - ' +patient_plane_dict[patient_plane_determiner_str]+').html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path)


def box_annotator_func_for_contour_plots(fig, dataframe, x_val, y_val, text_prefix):
    optimal_dil_string_list = [text_prefix]
    for index, row in dataframe.iterrows():
        optimal_dil_x_coord = round(row['Test location (Prostate centroid origin) (X)'],1)
        optimal_dil_y_coord = round(row['Test location (Prostate centroid origin) (Y)'],1)
        optimal_dil_z_coord = round(row['Test location (Prostate centroid origin) (Z)'],1)
        if optimal_dil_x_coord < 0:
            optimal_dil_x_coord_string = str(optimal_dil_x_coord) + ' R'
        else: 
            optimal_dil_x_coord_string = str(optimal_dil_x_coord) + ' L'
        if optimal_dil_y_coord < 0:
            optimal_dil_y_coord_string = str(optimal_dil_y_coord) + ' A'
        else: 
            optimal_dil_y_coord_string = str(optimal_dil_y_coord) + ' P'
        if optimal_dil_z_coord < 0:
            optimal_dil_z_coord_string = str(optimal_dil_z_coord) + ' I'
        else: 
            optimal_dil_z_coord_string = str(optimal_dil_z_coord) + ' S'

        optimal_dil_coord_string = '('  +optimal_dil_x_coord_string+','+optimal_dil_y_coord_string+','+optimal_dil_z_coord_string+ ')'
        optimal_dil_string_temp = row["Relative DIL ID"]+ ': ' + optimal_dil_coord_string
        optimal_dil_string_list.append(optimal_dil_string_temp)

    optimal_dil_string = ' | '.join(optimal_dil_string_list)
    fig.add_annotation(x=x_val,
                        y=y_val,
                        xref="paper",
                        yref="paper",
                        text=optimal_dil_string,
                        showarrow=False,
                        font=dict(
                            family="Courier New, monospace",
                            size=16,
                            color="#000000"
                            ),
                        bordercolor="#000000",
                        borderwidth=2,
                        borderpad=4,
                        bgcolor="#ffffff",
                        opacity=1
                        )

    return fig


def production_plot_guidance_maps_max_planes(patientUID,
                                            patient_sp_output_figures_dir,
                                            pydicom_item,
                                            dil_ref,
                                            oar_ref,
                                            all_ref_key,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string
                                            ):
    

    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]                 

    specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
    selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]
    prostate_structure_index = selected_prostate_info["Index number"]
    prostate_found_bool = selected_prostate_info["Struct found bool"]


    if prostate_found_bool == True:
        selected_prostate_centroid = pydicom_item[oar_ref][prostate_structure_index]["Structure global centroid"].reshape(3)
    else: 
        selected_prostate_centroid = np.array([0,0,0])
    

    column_to_index_dict = {'Test location (Prostate centroid origin) (X)': 0, 
                            'Test location (Prostate centroid origin) (Y)': 1, 
                            'Test location (Prostate centroid origin) (Z)': 2
                            }

    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):    
        specific_dil_structureID = specific_dil_structure["ROI"]
        sp_dil_guidance_map_max_planes_dataframe = specific_dil_structure["Biopsy optimization: guidance map max-planes dataframe"]
        sp_dil_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"]
        #selected_prostate_info_and_centroid_dict = specific_dil_structure["Biopsy optimization: selected relative prostate dict"]
        #selected_prostate_centroid = selected_prostate_info_and_centroid_dict["Centroid vector array"]
        
        

        list_of_max_plane_types = sp_dil_guidance_map_max_planes_dataframe['Patient plane'].unique().tolist()
        for plane in list_of_max_plane_types:
            plane_specific_guidance_map_max_planes_dataframe = sp_dil_guidance_map_max_planes_dataframe[sp_dil_guidance_map_max_planes_dataframe['Patient plane'] == plane]
            hor_axis_column_name = plane_specific_guidance_map_max_planes_dataframe.sample().reset_index(drop=True).at[0,'Coord 1 name']
            vert_axis_column_name = plane_specific_guidance_map_max_planes_dataframe.sample().reset_index(drop=True).at[0,'Coord 2 name']
            const_axis_column_name = plane_specific_guidance_map_max_planes_dataframe.sample().reset_index(drop=True).at[0,'Const coord name']
            
            fig = go.Figure()

            # Add optimal sampling location for each DIL
            for index, row in sp_dil_optimal_locations_dataframe.iterrows():
                fig.add_scatter(x=[row[hor_axis_column_name]],
                        y=[row[vert_axis_column_name]],
                        marker=dict(
                            color='orange',
                            size=10,
                            symbol = 'circle'
                        ),
                        text=[row["Relative DIL ID"]],
                        mode = "markers+text",
                        name=row["Relative DIL ID"]+' optimal',
                        textposition="bottom center",
                        textfont=dict(
                            family="sans serif",
                            size=12,
                            color="white"
                        )
                    )    


            # Add prostate centroid
            fig.add_scatter(x=[0],
                        y=[0],
                        marker=dict(
                            color='black',
                            size=12,
                            symbol = 'circle'
                        ),
                        mode = 'markers',
                    name='Prostate centroid')

            # need these two lines for both for loops (oars and dils)
            const_plane_coord_value = sp_dil_optimal_locations_dataframe.at[0,const_axis_column_name]
            const_plane_coord_value_in_patient_frame = const_plane_coord_value + selected_prostate_centroid[column_to_index_dict[const_axis_column_name]]
            for oar_structure in pydicom_item[oar_ref]:
                oar_structureID = oar_structure["ROI"]
                interslice_interpolation_information = oar_structure["Inter-slice interpolation information"]                        

                interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                #zslices_list = interslice_interpolation_information.interpolated_pts_list

                intraslice_interpolation_information = oar_structure["Intra-slice interpolation information"]

                intraslice_interpolated_zslices_list = intraslice_interpolation_information.interpolated_pts_list

                if 'Transverse' in plane:
                    # check if this structure is not in this plane
                    if (const_plane_coord_value_in_patient_frame > max(interpolated_zvals_list)) or (const_plane_coord_value_in_patient_frame < min(interpolated_zvals_list)):
                        continue
                    closest_z_index, closest_z_val = point_containment_tools.take_closest_numpy(interpolated_zvals_list, [const_plane_coord_value_in_patient_frame])
                    transverse_slice_of_oar_at_optimal_depth = np.array(intraslice_interpolated_zslices_list[closest_z_index[0]])
                    transverse_slice_of_oar_at_optimal_depth_in_prostate_frame = transverse_slice_of_oar_at_optimal_depth - selected_prostate_centroid
                    transverse_slice_of_oar_at_optimal_depth_in_prostate_frame_with_first_point_appended_for_connection = np.append(transverse_slice_of_oar_at_optimal_depth_in_prostate_frame, transverse_slice_of_oar_at_optimal_depth_in_prostate_frame[[0],:], axis = 0)

                    fig.add_scatter(x=transverse_slice_of_oar_at_optimal_depth_in_prostate_frame_with_first_point_appended_for_connection[:,column_to_index_dict[hor_axis_column_name]],
                        y=transverse_slice_of_oar_at_optimal_depth_in_prostate_frame_with_first_point_appended_for_connection[:,column_to_index_dict[vert_axis_column_name]],
                        mode = "lines",
                        name=oar_structureID)

                if ('Sagittal' in plane) or ('Coronal' in plane):
                    list_of_columns = list(column_to_index_dict.keys())
                    list_of_columns.remove('Test location (Prostate centroid origin) (Z)')
                    list_of_columns.remove(const_axis_column_name)
                    non_z_axis_non_const_axis_column_name = list_of_columns[0]
                    
                    # check if this structure is not in this plane
                    if (const_plane_coord_value_in_patient_frame > max(interpolated_pts_np_arr[:,column_to_index_dict[const_axis_column_name]])) or (const_plane_coord_value_in_patient_frame < min(interpolated_pts_np_arr[:,column_to_index_dict[const_axis_column_name]])):
                        continue
                    tolerance = 1
                    const_saggital_arr_above = np.empty((0,3), float)
                    const_saggital_arr_below = np.empty((0,3), float)
                    for const_slice_arr in intraslice_interpolated_zslices_list:
                        found_above = False
                        found_below = False
                        const_slice_centroid = centroid_finder.centeroidfinder_numpy_3D(const_slice_arr)
                        const_slice_centroid_vert_val = const_slice_centroid[0,column_to_index_dict[non_z_axis_non_const_axis_column_name]]
                        perp_distances_arr = abs(const_slice_arr[:,column_to_index_dict[const_axis_column_name]] - const_plane_coord_value_in_patient_frame)
                        nearest_points_indices_sorted = np.argsort(perp_distances_arr)
                        for pt_index in nearest_points_indices_sorted:
                            if (found_above == True) and (found_below == True):
                                break
                            test_pt = const_slice_arr[pt_index]
                            test_pt_vert_val = test_pt[column_to_index_dict[non_z_axis_non_const_axis_column_name]]
                            test_pt_in_prostate_frame = test_pt - selected_prostate_centroid
                            if (test_pt_vert_val > const_slice_centroid_vert_val) and (found_above == False) and (perp_distances_arr[pt_index] < tolerance):
                                const_saggital_arr_above = np.vstack([const_saggital_arr_above,test_pt_in_prostate_frame])
                                found_above = True
                            elif (test_pt_vert_val < const_slice_centroid_vert_val) and (found_below == False) and (perp_distances_arr[pt_index] < tolerance):
                                const_saggital_arr_below = np.vstack([const_saggital_arr_below,test_pt_in_prostate_frame])
                                found_below = True
                            
                    const_saggital_arr_below_ref = np.flip(const_saggital_arr_below, axis = 0)
                    full_sagg_slice_arr = np.vstack([const_saggital_arr_above, const_saggital_arr_below_ref])
                    full_sagg_slice_arr_with_first_point_appended_for_connection = np.vstack([full_sagg_slice_arr, full_sagg_slice_arr[0,:]])


                    fig.add_scatter(x=full_sagg_slice_arr_with_first_point_appended_for_connection[:,column_to_index_dict[hor_axis_column_name]],
                        y=full_sagg_slice_arr_with_first_point_appended_for_connection[:,column_to_index_dict[vert_axis_column_name]],
                        mode = "lines",
                        name=oar_structureID)


                    
            for dil_structure in pydicom_item[dil_ref]:
                dil_structureID = dil_structure["ROI"]
                interslice_interpolation_information = dil_structure["Inter-slice interpolation information"]                        

                interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                #zslices_list = interslice_interpolation_information.interpolated_pts_list


                intraslice_interpolation_information = dil_structure["Intra-slice interpolation information"]

                intraslice_interpolated_zslices_list = intraslice_interpolation_information.interpolated_pts_list


                if 'Transverse' in plane:
                    # check if this structure is not in this plane
                    if (const_plane_coord_value_in_patient_frame > max(interpolated_zvals_list)) or (const_plane_coord_value_in_patient_frame < min(interpolated_zvals_list)):
                        continue
                    closest_z_index, closest_z_val = point_containment_tools.take_closest_numpy(interpolated_zvals_list, [const_plane_coord_value_in_patient_frame])
                    transverse_slice_of_dil_at_optimal_depth = np.array(intraslice_interpolated_zslices_list[closest_z_index[0]])
                    transverse_slice_of_dil_at_optimal_depth_in_prostate_frame = transverse_slice_of_dil_at_optimal_depth - selected_prostate_centroid
                    transverse_slice_of_dil_at_optimal_depth_in_prostate_frame_with_first_point_appended_for_connection = np.append(transverse_slice_of_dil_at_optimal_depth_in_prostate_frame, transverse_slice_of_dil_at_optimal_depth_in_prostate_frame[[0],:], axis = 0)

                    fig.add_scatter(x=transverse_slice_of_dil_at_optimal_depth_in_prostate_frame_with_first_point_appended_for_connection[:,column_to_index_dict[hor_axis_column_name]],
                        y=transverse_slice_of_dil_at_optimal_depth_in_prostate_frame_with_first_point_appended_for_connection[:,column_to_index_dict[vert_axis_column_name]],
                        mode = "lines",
                        name=dil_structureID)
                    

                if ('Sagittal' in plane) or ('Coronal' in plane):
                    list_of_columns = list(column_to_index_dict.keys())
                    list_of_columns.remove('Test location (Prostate centroid origin) (Z)')
                    list_of_columns.remove(const_axis_column_name)
                    non_z_axis_non_const_axis_column_name = list_of_columns[0]
                    
                    # check if this structure is not in this plane
                    if (const_plane_coord_value_in_patient_frame > max(interpolated_pts_np_arr[:,column_to_index_dict[const_axis_column_name]])) or (const_plane_coord_value_in_patient_frame < min(interpolated_pts_np_arr[:,column_to_index_dict[const_axis_column_name]])):
                        continue
                    tolerance = 1
                    const_saggital_arr_above = np.empty((0,3), float)
                    const_saggital_arr_below = np.empty((0,3), float)
                    for const_slice_arr in intraslice_interpolated_zslices_list:
                        found_above = False
                        found_below = False
                        const_slice_centroid = centroid_finder.centeroidfinder_numpy_3D(const_slice_arr)
                        const_slice_centroid_vert_val = const_slice_centroid[0,column_to_index_dict[non_z_axis_non_const_axis_column_name]]
                        perp_distances_arr = abs(const_slice_arr[:,column_to_index_dict[const_axis_column_name]] - const_plane_coord_value_in_patient_frame)
                        nearest_points_indices_sorted = np.argsort(perp_distances_arr)
                        for pt_index in nearest_points_indices_sorted:
                            if (found_above == True) and (found_below == True):
                                break
                            test_pt = const_slice_arr[pt_index]
                            test_pt_vert_val = test_pt[column_to_index_dict[non_z_axis_non_const_axis_column_name]]
                            test_pt_in_prostate_frame = test_pt - selected_prostate_centroid

                            if (test_pt_vert_val > const_slice_centroid_vert_val) and (found_above == False) and (perp_distances_arr[pt_index] < tolerance):
                                const_saggital_arr_above = np.vstack([const_saggital_arr_above,test_pt_in_prostate_frame])
                                found_above = True
                            elif (test_pt_vert_val < const_slice_centroid_vert_val) and (found_below == False) and (perp_distances_arr[pt_index] < tolerance):
                                const_saggital_arr_below = np.vstack([const_saggital_arr_below,test_pt_in_prostate_frame])
                                found_below = True
                            
                    const_saggital_arr_below_ref = np.flip(const_saggital_arr_below, axis = 0)
                    full_sagg_slice_arr = np.vstack([const_saggital_arr_above, const_saggital_arr_below_ref])
                    full_sagg_slice_arr_with_first_point_appended_for_connection = np.vstack([full_sagg_slice_arr, full_sagg_slice_arr[0,:]])

                    fig.add_scatter(x=full_sagg_slice_arr_with_first_point_appended_for_connection[:,column_to_index_dict[hor_axis_column_name]],
                        y=full_sagg_slice_arr_with_first_point_appended_for_connection[:,column_to_index_dict[vert_axis_column_name]],
                        mode = "lines",
                        name=dil_structureID)

            # Add contour plot of the values of the cumulative projection of the optimization
            fig.add_trace(
                go.Contour(
                    z=plane_specific_guidance_map_max_planes_dataframe['Proportion of normal dist points contained'],
                    x=plane_specific_guidance_map_max_planes_dataframe[hor_axis_column_name],
                    y=plane_specific_guidance_map_max_planes_dataframe[vert_axis_column_name],
                    colorscale=[[0, 'rgb(0,0,255)'], [0.9, 'rgb(255,0,0)'],[1, 'rgb(0,255,0)']],
                    zmax = 1,
                    zmin = 0,
                    autocontour = False,
                    contours = go.contour.Contours(type = 'levels', showlines = True, coloring = 'heatmap', showlabels = True, size = 0.1),
                    connectgaps = False, 
                    colorbar = go.contour.ColorBar(len = 0.5)
                ))

            # Add axes names and other annotations on the figure
            x_axis_name = hor_axis_column_name[-2]
            y_axis_name = vert_axis_column_name[-2]
            patient_pos_dict = {'X': ' (L/R)', "Y":' (A/P)', "Z": '(S/I)'}
            fig['layout']['xaxis'].update(title=x_axis_name+patient_pos_dict[x_axis_name])
            fig['layout']['yaxis'].update(title=y_axis_name+patient_pos_dict[y_axis_name])
            
            patient_plane_dict = {'XY': ' Transverse (XY)', "YZ": ' Sagittal (YZ)', "XZ": ' Coronal (XZ)',
                                'YX': ' Transverse (YX)', "ZY": ' Sagittal (ZY)', "ZX": ' Coronal (ZX)'}
            patient_plane_determiner_str = x_axis_name+y_axis_name
            
            # Add annotation 
            text_1 = "Patient: "+ patientUID
            text_2 = "Optimal plane, "+patient_plane_dict[patient_plane_determiner_str]
            text_3 = 'Prostate centroid origin (mm)'
            text_list_for_annotation = [text_1,text_2,text_3]
            fig_description_text_for_annotation = ' | '.join(text_list_for_annotation)
            fig.add_annotation(text=fig_description_text_for_annotation,
                                    xref="paper", 
                                    yref="paper",
                                    x=0.99, 
                                    y=1.1, 
                                    showarrow=False,
                                    font=dict(family="Courier New, monospace", 
                                                size=16, 
                                                color="#000000"),
                                    bordercolor="#000000",
                                    borderwidth=2,
                                    borderpad=4,
                                    bgcolor="#ffffff",
                                    opacity=1
                                    )


            # add annotations for the optimal positions
            fig = box_annotator_func_for_contour_plots(fig, sp_dil_optimal_locations_dataframe, 0.99, 1.06, 'Optimal: ')

            
            svg_dose_fig_name = general_plot_name_string+' - (maxplane - ' +patient_plane_dict[patient_plane_determiner_str]+' - '+specific_dil_structureID+').svg'
            svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
            fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

            html_dose_fig_name = general_plot_name_string+' - (maxplane - ' +patient_plane_dict[patient_plane_determiner_str]+' - '+specific_dil_structureID+').html'
            html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
            fig.write_html(html_dose_fig_file_path)






def guidance_map_transducer_angle_sagittal(patientUID,
                                            patient_sp_output_figures_dir,
                                            pydicom_item,
                                            dil_ref,
                                            oar_ref,
                                            rectum_ref,
                                            all_ref_key,
                                            structs_referenced_dict,
                                            plot_open3d_structure_set_complete_demonstration_bool,
                                            biopsy_fire_travel_distances,
                                            biopsy_needle_compartment_length,
                                            important_info,
                                            live_display,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string
                                            ):
    
    
    contour_plot_list_of_dicts = advanced_guidance_map_creator.create_advanced_guidance_map_contour_plot(patientUID,
                                                                                    pydicom_item,
                                                                                    dil_ref,
                                                                                    all_ref_key,
                                                                                    oar_ref,
                                                                                    rectum_ref,
                                                                                    structs_referenced_dict,
                                                                                    plot_open3d_structure_set_complete_demonstration_bool,
                                                                                    biopsy_fire_travel_distances,
                                                                                    biopsy_needle_compartment_length,
                                                                                    important_info,
                                                                                    live_display,
                                                                                    transducer_plane_grid_spacing = 2)
    
    for contour_plot_dict in contour_plot_list_of_dicts:
        dil_id = contour_plot_dict["DIL ID"]
        contour_plot = contour_plot_dict["Contour plot"]
        
        svg_dose_fig_name = f"{dil_id} - {general_plot_name_string}.svg"
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        contour_plot.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = f"{dil_id} - {general_plot_name_string}.html"
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        contour_plot.write_html(html_dose_fig_file_path)


def guidance_map_transducer_angle_sagittal_and_max_plane_transverse(patientUID,
                                            patient_sp_output_figures_dir,
                                            pydicom_item,
                                            dil_ref,
                                            oar_ref,
                                            rectum_ref,
                                            all_ref_key,
                                            structs_referenced_dict,
                                            plot_open3d_structure_set_complete_demonstration_bool,
                                            biopsy_fire_travel_distances,
                                            biopsy_needle_compartment_length,
                                            interp_inter_slice_dist,
                                            interp_intra_slice_dist,
                                            radius_for_normals_estimation,
                                            max_nn_for_normals_estimation,
                                            important_info,
                                            live_display,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string
                                            ):
    
    
    trus_plane_sagittal_contour_plot_list_of_dicts, transverse_contour_plot_list_of_dicts = advanced_guidance_map_creator.create_advanced_guidance_map_transducer_saggital_and_transverse_contour_plot(patientUID,
                                                                                    pydicom_item,
                                                                                    dil_ref,
                                                                                    all_ref_key,
                                                                                    oar_ref,
                                                                                    rectum_ref,
                                                                                    structs_referenced_dict,
                                                                                    plot_open3d_structure_set_complete_demonstration_bool,
                                                                                    biopsy_fire_travel_distances,
                                                                                    biopsy_needle_compartment_length,
                                                                                    interp_inter_slice_dist,
                                                                                    interp_intra_slice_dist,
                                                                                    radius_for_normals_estimation,
                                                                                    max_nn_for_normals_estimation,
                                                                                    important_info,
                                                                                    live_display,
                                                                                    transducer_plane_grid_spacing = 2)
    
    for contour_plot_dict in trus_plane_sagittal_contour_plot_list_of_dicts:
        dil_id = contour_plot_dict["DIL ID"]
        contour_plot = contour_plot_dict["Contour plot"]
        
        svg_dose_fig_name = f"{dil_id} - {general_plot_name_string} - TRUS plane MAX sagittal.svg"
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        contour_plot.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = f"{dil_id} - {general_plot_name_string} - TRUS plane MAX sagittal.html"
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        contour_plot.write_html(html_dose_fig_file_path)

    for contour_plot_dict in transverse_contour_plot_list_of_dicts:
        dil_id = contour_plot_dict["DIL ID"]
        contour_plot = contour_plot_dict["Contour plot"]
        
        svg_dose_fig_name = f"{dil_id} - {general_plot_name_string} - Transverse plane MAX.svg"
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        contour_plot.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

        html_dose_fig_name = f"{dil_id} - {general_plot_name_string} - Transverse plane MAX.html"
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        contour_plot.write_html(html_dose_fig_file_path)




"""
def production_plot_cohort_biopsy_maps_dil_coord_frame(patientUID,
                            patient_sp_output_figures_dir,
                            pydicom_item,
                            all_ref_key,
                            oar_ref,
                            svg_image_scale,
                            svg_image_width,
                            svg_image_height,
                            general_plot_name_string
                            ):


    sp_patient_relative_dil_dataframe = pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Nearest DILs info dataframe"]


    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
            x = x,
            y = y,
            colorscale = 'Blues',
            reversescale = True,
            xaxis = 'x',
            yaxis = 'y'
        ))
    fig.add_trace(go.Scatter(
            x = x,
            y = y,
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = 'rgba(0,0,0,0.3)',
                size = 3
            )
        ))
    fig.add_trace(go.Histogram(
            y = y,
            xaxis = 'x2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))
    fig.add_trace(go.Histogram(
            x = x,
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))

    fig.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        height = 600,
        width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False
    )

    fig.show()



"""




def scatter_matrix_dil_features_cohort(master_cohort_patient_data_and_dataframes,
                                       dil_ref):

    structure_cohort_3d_radiomic_features_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: 3D radiomic features all OAR and DIL structures"]

    dils_only_structure_cohort_3d_radiomic_features_dataframe = structure_cohort_3d_radiomic_features_dataframe[structure_cohort_3d_radiomic_features_dataframe["Structure type"] == dil_ref]

    list_of_feature_columns = ["Volume",
     "Surface area",
     "Surface area to volume ratio",
     "Sphericity",
     "Compactness 1",
     "Compactness 2", 
     "Spherical disproportion",
     "Maximum 3D diameter",
     "PCA major",
     "PCA minor",
     "PCA least",
     "Major axis (equivalent ellipse)",
     "Minor axis (equivalent ellipse)",
     "Least axis (equivalent ellipse)",
     "Elongation",
     "Flatness",
     "L/R dimension at centroid",
     "A/P dimension at centroid",
     "S/I dimension at centroid"]
    
    dils_only_structure_cohort_3d_radiomic_features_dataframe_only_features = dils_only_structure_cohort_3d_radiomic_features_dataframe[list_of_feature_columns]
    
    



def global_tissue_class_target_dil_score_versus_target_3d_segmentation_radiomic_features(cohort_global_tissue_scores_with_target_dil_radiomic_features_df):


    # Assuming 'df' is your DataFrame
    features = [
        "Volume",
        "Surface area",
        "Surface area to volume ratio",
        "Sphericity",
        "Compactness 1",
        "Compactness 2",
        "Spherical disproportion",
        "Maximum 3D diameter",
        "PCA major",
        "PCA minor",
        "PCA least",
        "Major axis (equivalent ellipse)",
        "Minor axis (equivalent ellipse)",
        "Least axis (equivalent ellipse)",
        "Elongation",
        "Flatness",
        "L/R dimension at centroid",
        "A/P dimension at centroid",
        "S/I dimension at centroid"
    ]

    for feature in features:
        sns.lmplot(x=feature, y="Global mean binom est", hue="Simulated type", data=cohort_global_tissue_scores_with_target_dil_radiomic_features_df, 
                aspect=1.5, ci=None, palette="Set1")
        plt.title(f'Global mean binom est vs. {feature} by Simulated type')
        plt.xlabel(feature)
        plt.ylabel("Global mean binom est")
        plt.show()



def production_plot_binom_est_ridge_plot_by_voxel(master_cohort_patient_data_and_dataframes,
                                             svg_image_width,
                                             svg_image_height,
                                             dpi,
                                             ridge_line_tissue_class_general_plot_name_string,
                                            patient_sp_output_figures_dir_dict):

    plt.ioff()
    cohort_all_dose_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise binom est distribution"]
    df = cohort_all_dose_data_by_trial_and_pt 

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    #output_directory = Path(output_directory_path)
    #output_directory.mkdir(parents=True, exist_ok=True)

    dpi = 100
    figure_width_in = 1920 / dpi
    figure_height_in = 1080 / dpi

    for (patient_id, bx_index, structure_roi), group in df.groupby(['Patient ID', 'Bx index', 'Structure ROI']):
        bx_structure_roi = group['Bx structure ROI'].iloc[0]

        # Ensure the column is in the correct numerical format
        group["Mean probability (binom est)"] = pandas.to_numeric(group["Mean probability (binom est)"], errors='coerce')
        group.dropna(subset=["Mean probability (binom est)"], inplace=True)

        num_voxels = len(group['Voxel index'].unique())
        pal = sns.color_palette("husl", num_voxels)

        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=figure_height_in / num_voxels, palette=pal)

        g.map(sns.kdeplot, "Mean probability (binom est)", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "Mean probability (binom est)", clip_on=False, color="w", lw=2, bw_adjust=0.5)
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        for ax in g.axes.flat:
            ax.set_xlim(0, 1)

        def annotate_stats(x, color, label):
            label_float = float(label)
            specific_group = group[group['Voxel index'] == label_float]

            if not specific_group.empty:
                mean_prob = np.mean(x)
                std_prob = np.std(x)
                if np.std(x) < 1e-6 or len(np.unique(x)) <= 1:
                    # Data lacks variability, skip KDE and use placeholders for max density value
                    max_density_value = "N/A"  # Placeholder
                else:
                    try:
                        kde = gaussian_kde(x)
                        x_grid = np.linspace(0, 1, 1000)
                        max_density_value = x_grid[np.argmax(kde(x_grid))]
                        max_density_value = f"{max_density_value:.2f}"
                    except np.linalg.LinAlgError:
                        max_density_value = "Error"  # In case of an unexpected error
                
                

                nominal_mean = specific_group['Nominal containment'].mean()
                nominal_std = specific_group['Nominal containment'].std()

                voxel_begin = specific_group['Voxel begin (Z)'].iloc[0]
                voxel_end = specific_group['Voxel end (Z)'].iloc[0]

                ax = plt.gca()
                annotation_text = f'({voxel_begin}, {voxel_end})\nMean: {mean_prob:.2f}, SD: {std_prob:.2f}, Max Density Value: {max_density_value}\nNominal Mean: {nominal_mean:.2f}, Nominal STD: {nominal_std:.2f}'
                ax.text(1, 0.3, annotation_text, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        g.map(annotate_stats, "Mean probability (binom est)")

        g.figure.subplots_adjust(hspace=-0.25)
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)

        g.set_axis_labels("Mean probability", "")


        g.fig.set_size_inches(figure_width_in, figure_height_in)
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)


        dpi = 100  # DPI setting for saving the image

        # Calculate the figure size in inches for the desired dimensions in pixels
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        # Title now includes Structure ROI
        plt.suptitle(f'Patient ID: {patient_id}, Bx structure ROI: {bx_structure_roi}, Structure ROI: {structure_roi}', fontsize=16, fontweight='bold', y=0.95)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_tissue_class_general_plot_name_string+str(bx_structure_roi)+ str(structure_roi)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)


def production_plot_binom_est_ridge_plot_by_voxel_v2(master_cohort_patient_data_and_dataframes,
                                             svg_image_width,
                                             svg_image_height,
                                             dpi,
                                             ridge_line_tissue_class_general_plot_name_string,
                                            patient_sp_output_figures_dir_dict,
                                            cancer_tissue_label):

    plt.ioff()
    cohort_all_binom_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise binom est distribution"]
    df = cohort_all_binom_data_by_trial_and_pt[cohort_all_binom_data_by_trial_and_pt["Structure ROI"] == cancer_tissue_label]

    # Define the colormap with a more scientific approach
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    dirac_delta_color = plt.cm.RdYlGn(0.01)

    # Set the theme for seaborn plots
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


    # density plotting, coloring and annotator function
    def annotate_and_color(x, color, label, **kwargs):
        label_float = float(label)
        specific_group = group[group['Voxel index'] == label_float]

        if not specific_group.empty:
            mean_prob = np.mean(x)
            std_prob = np.std(x)
            nominal_mean = specific_group['Nominal containment'].mean()
            nominal_std = specific_group['Nominal containment'].std()

            ax = plt.gca()

            if np.std(x) < 1e-6 or len(np.unique(x)) <= 1:
                # Data lacks variability, skip KDE and use placeholders for max density value
                if np.all(x == 0):
                    # If all values are zero, plot a spike at 0
                    ax.axvline(x=0, color=dirac_delta_color, linestyle='-', lw=2)
                    max_density_value = 0
                else:
                    max_density_value = mean_prob  # If there is no variability then the argmax should just be the mean
            else:
                try:
                    kde = gaussian_kde(x)
                    x_grid = np.linspace(0, 1, 1000)
                    max_density_value = x_grid[np.argmax(kde(x_grid))]
                    max_density_value = f"{max_density_value:.2f}"

                    density_color = cmap(norm(nominal_mean))

                    y_density = kde(x_grid)
                    
                    # Find the maximum y-density value and calculate the scaling factor
                    max_density = np.max(y_density)
                    scaling_factor = 1.0 / max_density if max_density > 0 else 1
                    
                    # Scale the density values
                    scaled_density = y_density * scaling_factor
                    
                    # Plot the scaled density as filled area and overlay with a black ridgeline
                    ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)
                except np.linalg.LinAlgError:
                    max_density_value = "Error"  # In case of an unexpected error
            
        

            voxel_begin = specific_group['Voxel begin (Z)'].iloc[0]
            voxel_end = specific_group['Voxel end (Z)'].iloc[0]

            
            annotation_text = f'Tissue segment (mm): ({voxel_begin}, {voxel_end})\nMean: {mean_prob:.2f} | SD: {std_prob:.2f} | argmax(Density): {max_density_value}\nNominal Mean: {nominal_mean:.2f} | Nominal SD: {nominal_std:.2f}'
            ax.text(0.98, 0.6, annotation_text, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        
    # Main loop for plotting
    for (patient_id, bx_index), group in df.groupby(['Patient ID', 'Bx index']):
        unique_voxels = group['Voxel index'].unique()
        palette_black = {voxel: "black" for voxel in unique_voxels}

        bx_id = group['Bx structure ROI'].iloc[0]

        # Setup for each subplot (FacetGrid)
        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)
        g.map(annotate_and_color, "Mean probability (binom est)")


        # Adjust layout to make room for the colorbar
        g.fig.subplots_adjust(right=0.85, left= 0.07)
        cbar_ax = g.fig.add_axes([0.9, 0.1, 0.05, 0.7])
        g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # Final adjustments
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=False, left=True)
        g.set_axis_labels("Probability", "")
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)

        # Title and figure size
        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        dpi = 100  # DPI setting for saving the image

        # Calculate the figure size in inches for the desired dimensions in pixels
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_tissue_class_general_plot_name_string+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)



def production_plot_dose_ridge_plot_by_voxel(master_cohort_patient_data_and_dataframes,
                                             svg_image_width,
                                             svg_image_height,
                                             dpi,
                                             ridge_line_dose_general_plot_name_string,
                                            patient_sp_output_figures_dir_dict
                                            ):
    plt.ioff()
    cohort_all_dose_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise dose distribution"]
    df = cohort_all_dose_data_by_trial_and_pt 

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    for (patient_id, bx_index), group in df.groupby(['Patient ID', 'Bx index']):
        bx_id = group['Bx ID'].iloc[0]

        num_voxels = len(group['Voxel index'].unique())
        pal = sns.color_palette("husl", num_voxels)

        ridge_height = 1
        figure_height = num_voxels * ridge_height + 1

        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=ridge_height, palette=pal)

        g.map(sns.kdeplot, "Dose (Gy)", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "Dose (Gy)", clip_on=False, color="w", lw=2, bw_adjust=0.5)
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        def annotate_stats(x, color, label, **kwargs):
            label_float = float(label)
            voxel_row = group[(group['Voxel index'] == label_float) & (group['MC trial num'] == 0)].iloc[0]
            nominal_dose = voxel_row['Dose (Gy)'] if not voxel_row.empty else 'N/A'

            mean = np.mean(x)
            std = np.std(x)
            kde = gaussian_kde(x)
            x_grid = np.linspace(x.min(), x.max(), 1000)
            max_density_dose = x_grid[np.argmax(kde(x_grid))]
            voxel_begin = voxel_row['Voxel begin (Z)']
            voxel_end = voxel_row['Voxel end (Z)']
            ax = plt.gca()
            annotation_text = f'Voxel position ({voxel_begin}, {voxel_end})\nMean: {mean:.2f}, SD: {std:.2f}, argmax(Density): {max_density_dose:.2f}, Nominal: {nominal_dose:.2f}'
            ax.text(0.95, 0.5, annotation_text, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)
    
        g.map(annotate_stats, "Dose (Gy)")

        g.figure.subplots_adjust(hspace=-0.25)
        g.set_titles("")
        g.set(yticks=[],)
        g.despine(bottom=True, left=True)

        g.set_axis_labels("Dose (Gy)", "")

        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)



        dpi = 100  # DPI setting for saving the image

        # Calculate the figure size in inches for the desired dimensions in pixels
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        
        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_dose_general_plot_name_string+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)


def production_plot_dose_ridge_plot_by_voxel_v2(master_cohort_patient_data_and_dataframes,
                                             svg_image_width,
                                             svg_image_height,
                                             dpi,
                                             ridge_line_dose_general_plot_name_string,
                                            patient_sp_output_figures_dir_dict
                                            ):
    plt.ioff()
    cohort_all_dose_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise dose distribution"]
    df = cohort_all_dose_data_by_trial_and_pt 


    # Define the colormap with a more scientific approach
    cmap = plt.cm.turbo
    norm = Normalize(vmin=13.5, vmax=30)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Set the theme for seaborn plots
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    

    # Define functions outside the loop
    def annotate_and_color(x, color, label, **kwargs):
        label_float = float(label)
        voxel_row = group[(group['Voxel index'] == label_float) & (group['MC trial num'] == 0)].iloc[0]
        nominal_dose = voxel_row['Dose (Gy)'] if not voxel_row.empty else 'N/A'

        mean = np.mean(x)
        std = np.std(x)
        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 1000)
        max_density_dose = x_grid[np.argmax(kde(x_grid))]
        voxel_begin = voxel_row['Voxel begin (Z)']
        voxel_end = voxel_row['Voxel end (Z)']
        ax = plt.gca()
        annotation_text = f'Tissue segment (mm): ({voxel_begin}, {voxel_end})\nMean (Gy): {mean:.2f} | SD (Gy): {std:.2f} | argmax(Density) (Gy): {max_density_dose:.2f} | Nominal (Gy): {nominal_dose:.2f}'
        ax.text(0.98, 0.7, annotation_text, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        rounded_max_dose = np.clip(round(max_density_dose), 0, 50)
        density_color = cmap(norm(rounded_max_dose))

        y_density = kde(x_grid)
        
        # Find the maximum y-density value and calculate the scaling factor
        max_density = np.max(y_density)
        scaling_factor = 1.0 / max_density if max_density > 0 else 1
        
        # Scale the density values
        scaled_density = y_density * scaling_factor
        
        # Plot the scaled density as filled area and overlay with a black ridgeline
        ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)

        #sns.kdeplot(x, bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5, color=density_color, ax=ax)  # For fill color based on logic
        #sns.kdeplot(x, bw_adjust=0.5, clip_on=False, color="black", lw=2, ax=ax)  # For black ridgeline

    # Main loop for plotting
    for (patient_id, bx_index), group in df.groupby(['Patient ID', 'Bx index']):
        unique_voxels = group['Voxel index'].unique()
        palette_black = {voxel: "black" for voxel in unique_voxels}

        bx_id = group['Bx ID'].iloc[0]

        # Setup for each subplot (FacetGrid)
        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)
        g.map(annotate_and_color, "Dose (Gy)")

        # Adjust layout to make room for the colorbar
        g.fig.subplots_adjust(right=0.85, left= 0.07)
        cbar_ax = g.fig.add_axes([0.9, 0.1, 0.05, 0.7])
        g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # Final adjustments
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=False, left=True)
        g.set_axis_labels("Dose (Gy)", "")
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)

        # Title and figure size
        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        dpi = 100  # DPI setting for saving the image

        # Calculate the figure size in inches for the desired dimensions in pixels
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_dose_general_plot_name_string+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)


def production_plot_dose_ridge_plot_by_voxel_v3(multi_or_sp_patient_dose_df,
                                             svg_image_width,
                                             svg_image_height,
                                             dpi,
                                             ridge_line_dose_general_plot_name_string,
                                            patient_sp_output_figures_dir_dict,
                                            all_ref
                                            ):
    plt.ioff()
    #sp_patient_all_dose_data_by_trial_and_pt = pydicom_item[all_ref]["Dosimetry - All points and trials"]
    df = multi_or_sp_patient_dose_df

    # Define the colormap with a more scientific approach
    cmap = plt.cm.turbo
    norm = Normalize(vmin=13.5, vmax=30)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Set the theme for seaborn plots
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    

    # Define functions outside the loop
    def annotate_and_color(x, color, label, **kwargs):
        label_float = float(label)
        voxel_row = group[(group['Voxel index'] == label_float) & (group['MC trial num'] == 0)].iloc[0]
        nominal_dose = voxel_row['Dose (Gy)'] if not voxel_row.empty else 'N/A'

        mean = np.mean(x)
        std = np.std(x)
        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 1000)
        max_density_dose = x_grid[np.argmax(kde(x_grid))]
        voxel_begin = voxel_row['Voxel begin (Z)']
        voxel_end = voxel_row['Voxel end (Z)']
        ax = plt.gca()
        annotation_text = f'Tissue segment (mm): ({voxel_begin}, {voxel_end})\nMean (Gy): {mean:.2f} | SD (Gy): {std:.2f} | argmax(Density) (Gy): {max_density_dose:.2f} | Nominal (Gy): {nominal_dose:.2f}'
        ax.text(0.98, 0.7, annotation_text, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        rounded_max_dose = np.clip(round(max_density_dose), 0, 50)
        density_color = cmap(norm(rounded_max_dose))

        y_density = kde(x_grid)
        
        # Find the maximum y-density value and calculate the scaling factor
        max_density = np.max(y_density)
        scaling_factor = 1.0 / max_density if max_density > 0 else 1
        
        # Scale the density values
        scaled_density = y_density * scaling_factor
        
        # Plot the scaled density as filled area and overlay with a black ridgeline
        ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)

        #sns.kdeplot(x, bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5, color=density_color, ax=ax)  # For fill color based on logic
        #sns.kdeplot(x, bw_adjust=0.5, clip_on=False, color="black", lw=2, ax=ax)  # For black ridgeline

    # Main loop for plotting
    for (patient_id, bx_index), group in df.groupby(['Patient ID', 'Bx index']):
        unique_voxels = group['Voxel index'].unique()
        palette_black = {voxel: "black" for voxel in unique_voxels}

        bx_id = group['Bx ID'].iloc[0]

        # Setup for each subplot (FacetGrid)
        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)
        g.map(annotate_and_color, "Dose (Gy)")

        # Adjust layout to make room for the colorbar
        g.fig.subplots_adjust(right=0.85, left= 0.07)
        cbar_ax = g.fig.add_axes([0.9, 0.1, 0.05, 0.7])
        cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        # Add a label to the colorbar
        cbar.set_label('Dose (Gy)', fontsize=12, labelpad=10, rotation=270, color='black')


        # Final adjustments
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=False, left=True)
        g.set_axis_labels("Dose (Gy)", "")
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)

        # Title and figure size
        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        dpi = 100  # DPI setting for saving the image

        # Calculate the figure size in inches for the desired dimensions in pixels
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_dose_general_plot_name_string+' - '+str(patient_id)+' - '+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)


def production_plot_dose_ridge_plot_by_voxel_with_tissue_class_coloring(master_cohort_patient_data_and_dataframes,
                                             svg_image_width,
                                             svg_image_height,
                                             dpi,
                                             ridge_line_dose_and_binom_general_plot_name_string,
                                            patient_sp_output_figures_dir_dict,
                                            cancer_tissue_label
                                            ):
    plt.ioff()
    cohort_all_dose_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise dose distribution"]
    df_dose = cohort_all_dose_data_by_trial_and_pt 

    cohort_all_binom_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise binom est distribution"]
    df_tissue = cohort_all_binom_data_by_trial_and_pt[cohort_all_binom_data_by_trial_and_pt["Structure ROI"] == cancer_tissue_label]

    # Define the colormap with a more scientific approach
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Set the theme for seaborn plots
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


    # Define functions outside the loop
    def annotate_and_color(x, color, label, **kwargs):
        label_float = float(label)
        voxel_row = group[(group['Voxel index'] == label_float) & (group['MC trial num'] == 0)].iloc[0]
        nominal_dose = voxel_row['Dose (Gy)'] if not voxel_row.empty else 'N/A'

        mean = np.mean(x)
        std = np.std(x)
        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 1000)
        max_density_dose = x_grid[np.argmax(kde(x_grid))]
        voxel_begin = voxel_row['Voxel begin (Z)']
        voxel_end = voxel_row['Voxel end (Z)']
        ax = plt.gca()
        annotation_text = f'Tissue segment (mm): ({voxel_begin}, {voxel_end})\nMean (Gy): {mean:.2f} | SD (Gy): {std:.2f} | argmax(Density) (Gy): {max_density_dose:.2f} | Nominal (Gy): {nominal_dose:.2f}'
        ax.text(0.98, 0.7, annotation_text, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        # find the kde density color based on the tissue class dataframe
        tissue_voxel = df_tissue[(df_tissue['Voxel index'] == label_float) & 
                                    (df_tissue['Patient ID'] == patient_id) & 
                                    (df_tissue['Bx index'] == bx_index)]
        binom_mean = tissue_voxel["Mean probability (binom est)"].mean()

        density_color = cmap(norm(binom_mean))

        # Rescale and plot the densities
        y_density = kde(x_grid)
        
        # Find the maximum y-density value and calculate the scaling factor
        max_density = np.max(y_density)
        scaling_factor = 1.0 / max_density if max_density > 0 else 1
        
        # Scale the density values
        scaled_density = y_density * scaling_factor
        
        # Plot the scaled density as filled area and overlay with a black ridgeline
        ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)


    # Main loop for plotting
    for (patient_id, bx_index), group in df_dose.groupby(['Patient ID', 'Bx index']):
        bx_id = group['Bx ID'].iloc[0]

        unique_voxels = group['Voxel index'].unique()
        palette_black = {voxel: "black" for voxel in unique_voxels}

        # Setup for each subplot (FacetGrid)
        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)
        g.map(annotate_and_color, "Dose (Gy)", df_tissue = df_tissue, patient_id = patient_id, bx_index = bx_index)

        # Adjust layout to make room for the colorbar
        g.fig.subplots_adjust(right=0.85, left= 0.07)
        cbar_ax = g.fig.add_axes([0.9, 0.1, 0.05, 0.7])
        g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # Final adjustments
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=False, left=True)
        g.set_axis_labels("Dose (Gy)", "")
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)

        # Title and figure size
        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        dpi = 100  # DPI setting for saving the image

        # Calculate the figure size in inches for the desired dimensions in pixels
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_dose_and_binom_general_plot_name_string+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)

def production_plot_dose_ridge_plot_by_voxel_with_tissue_class_coloring_no_dose_cohort(sp_patient_dose_df,
                                                                                        sp_patient_binom_df,
                                                                                        svg_image_width,
                                                                                        svg_image_height,
                                                                                        dpi,
                                                                                        ridge_line_dose_and_binom_general_plot_name_string,
                                                                                        patient_sp_output_figures_dir_dict,
                                                                                        cancer_tissue_label
                                                                                        ):
    plt.ioff()
    #cohort_all_dose_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise dose distribution"]
    df_dose = sp_patient_dose_df

    #cohort_all_binom_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise binom est distribution"]
    #df_tissue = cohort_all_binom_data_by_trial_and_pt[cohort_all_binom_data_by_trial_and_pt["Structure ROI"] == cancer_tissue_label]
    df_tissue = sp_patient_binom_df[sp_patient_binom_df["Structure ROI"] == cancer_tissue_label]

    # Define the colormap with a more scientific approach
    #cmap = plt.cm.RdYlGn
    # Invert the colormap
    #cmap = cmap.reversed()

    # Define colors in the order you want: green, blue, red
    colors = ["green", "blue", "black"]  # Hex codes or RGB tuples can also be used

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list("GreenBlueRed", colors, N=10)  # N is the number of color bins

    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Set the theme for seaborn plots
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


    # Define functions outside the loop
    def annotate_and_color(x, color, label, **kwargs):
        label_float = float(label)
        voxel_row = group[(group['Voxel index'] == label_float) & (group['MC trial num'] == 0)].iloc[0]
        voxel_df = group[(group['Voxel index'] == label_float) & (group['MC trial num'] == 0)]
        nominal_dose = voxel_df['Dose (Gy)'].mean() if not voxel_row.empty else 'N/A'

        mean = np.mean(x)
        std = np.std(x)
        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 1000)
        max_density_dose = x_grid[np.argmax(kde(x_grid))]
        voxel_begin = voxel_row['Voxel begin (Z)']
        voxel_end = voxel_row['Voxel end (Z)']

        # find the kde density color based on the tissue class dataframe
        tissue_voxel = df_tissue[(df_tissue['Voxel index'] == label_float) & 
                                    (df_tissue['Patient ID'] == patient_id) & 
                                    (df_tissue['Bx index'] == bx_index)]
        binom_mean = tissue_voxel["Mean probability (binom est)"].mean()


        ax = plt.gca()
        annotation_text = f'Tissue segment (mm): ({voxel_begin:.1f}, {voxel_end:.1f}) | Tumor tissue score: {binom_mean:.2f}\nMean (Gy): {mean:.2f} | SD (Gy): {std:.2f} | argmax(Density) (Gy): {max_density_dose:.2f} | Nominal (Gy): {nominal_dose:.2f}'
        ax.text(1.03, 0.7, annotation_text, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        # Plot vertical lines
        ax.axvline(x=max_density_dose, color='magenta', linestyle='-', label='Max Density (Gy)')
        ax.axvline(x=mean, color='orange', linestyle='-', label='Mean (Gy)')
        ax.axvline(x=nominal_dose, color='red', linestyle='-', label='Nominal (Gy)')

        density_color = cmap(norm(binom_mean))

        # Rescale and plot the densities
        y_density = kde(x_grid)
        
        # Find the maximum y-density value and calculate the scaling factor
        max_density = np.max(y_density)
        scaling_factor = 1.0 / max_density if max_density > 0 else 1
        
        # Scale the density values
        scaled_density = y_density * scaling_factor
        
        # Plot the scaled density as filled area and overlay with a black ridgeline
        ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)


    # Calculate maximum x-axis limit from 95th quantile
    max_95th_quantile = df_dose.groupby('Voxel index')['Dose (Gy)'].quantile(0.95).max()

    # Calculate maximum x-axis limit from 95th quantile
    min_5th_quantile = df_dose.groupby('Voxel index')['Dose (Gy)'].quantile(0.05).min()

    # Define legend handles
    legend_handles = [
        Line2D([0], [0], color='magenta', lw=2, linestyle='-', label='Max Density (Gy)'),
        Line2D([0], [0], color='orange', lw=2, linestyle='-', label='Mean (Gy)'),
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='Nominal (Gy)')
    ]

    # Main loop for plotting
    for (patient_id, bx_index), group in df_dose.groupby(['Patient ID', 'Bx index']):
        #bx_id = group['Bx ID'].iloc[0]
        bx_id = group.at[0,'Bx ID']

        unique_voxels = group['Voxel index'].unique()
        palette_black = {voxel: "black" for voxel in unique_voxels}

        # Setup for each subplot (FacetGrid)
        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)
        g.map(annotate_and_color, "Dose (Gy)", df_tissue = df_tissue, patient_id = patient_id, bx_index = bx_index)

        # Set the x-axis limits uniformly
        g.set(xlim=(min_5th_quantile, max_95th_quantile))


        # Adjust layout to make room for the colorbar and text annotations
        g.fig.subplots_adjust(right=0.53, left= 0.07, top = 0.95, bottom = 0.05)
        cbar_ax = g.fig.add_axes([0.9, 0.2, 0.03, 0.6])
        g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        # Final adjustments
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=False, left=True)
        g.set_axis_labels("Dose (Gy)", "")
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)
        g.fig.text(0.88, 0.5, 'Tumor tissue score', va='center', rotation='vertical', fontsize=12)

        # Adjust subplot widths to 75% of figure
        for ax in g.axes.flat:
            #pos = ax.get_position()
            #ax.set_position([pos.x0, pos.y0, pos.width * 0.5, pos.height])
            
            # Setting major and minor ticks
            #ax.xaxis.set_major_locator(ticker.AutoLocator())  # Auto-set major ticks
            #ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))  # Set two minor ticks between major ticks

            # Optional: Customize tick appearance
            #ax.tick_params(axis='x', which='major', length=7, width=2, color='black', direction='inout')
            #ax.tick_params(axis='x', which='minor', length=4, color='gray')

            # Enable vertical grid for major and minor ticks
            ax.grid(True, which='both', axis='x', linestyle='-', color='gray', linewidth=0.5)
            ax.set_axisbelow(True)  # Ensures grid is behind plot elements

            #ax.xaxis.set_tick_params(which='both', labelbottom=True)
        #plt.grid(True)

        # Create the legend
        g.fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.9, 1), ncol=1, frameon=True, facecolor='white')


        # Title and figure size
        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        dpi = 100  # DPI setting for saving the image

        # Calculate the figure size in inches for the desired dimensions in pixels
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_dose_and_binom_general_plot_name_string+' - '+str(patient_id)+' - '+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)





def production_plot_dose_ridge_plot_by_voxel_with_tissue_class_coloring_no_dose_cohort_v2(sp_bx_dose_distribution_all_trials_df,
                                                                                          sp_patient_and_sp_bx_dose_dataframe_by_voxel,
                                                                                          sp_patient_binom_df,
                                                                                          svg_image_width,
                                                                                          svg_image_height,
                                                                                          dpi,
                                                                                          ridge_line_dose_and_binom_general_plot_name_string,
                                                                                          patient_sp_output_figures_dir_dict,
                                                                                          cancer_tissue_label):
    plt.ioff()
    

    df_dose = sp_bx_dose_distribution_all_trials_df
    df_dose = misc_tools.convert_categorical_columns(df_dose, ['Voxel index', "Dose (Gy)"], [int, float])

    df_dose_stats_by_voxel = sp_patient_and_sp_bx_dose_dataframe_by_voxel
    df_tissue = sp_patient_binom_df[sp_patient_binom_df["Structure ROI"] == cancer_tissue_label]

    colors = ["green", "blue", "black"]
    cmap = LinearSegmentedColormap.from_list("GreenBlueRed", colors, N=10)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    def annotate_and_color_v2(x, color, label, **kwargs):
        label_float = float(label)
        voxel_row = df_dose_stats_by_voxel[(df_dose_stats_by_voxel['Voxel index'] == label_float) & 
                           (df_dose_stats_by_voxel['Patient ID'] == patient_id) & 
                           (df_dose_stats_by_voxel['Bx index'] == bx_index)].iloc[0]

        nominal_dose = voxel_row[('Dose (Gy)','nominal')]
        mean = voxel_row[('Dose (Gy)','mean')]
        std = voxel_row[('Dose (Gy)','std')]
        max_density_dose = voxel_row[('Dose (Gy)','argmax_density')]
        voxel_begin = voxel_row[('Voxel begin (Z)', '')]
        voxel_end = voxel_row[('Voxel end (Z)', '')]

        q05_dose = voxel_row[('Dose (Gy)','quantile_05')]
        q25_dose = voxel_row[('Dose (Gy)','quantile_25')]
        q50_dose = voxel_row[('Dose (Gy)','quantile_50')]
        q75_dose = voxel_row[('Dose (Gy)','quantile_75')]
        q95_dose = voxel_row[('Dose (Gy)','quantile_95')]

        tissue_voxel = df_tissue[(df_tissue['Voxel index'] == label_float) & 
                                 (df_tissue['Patient ID'] == patient_id) & 
                                 (df_tissue['Bx index'] == bx_index)]
        binom_mean = tissue_voxel["Mean probability (binom est)"].mean()

        ax = plt.gca()
        annotation_text = f'Tissue segment (mm): ({voxel_begin:.1f}, {voxel_end:.1f}) | Tumor tissue score: {binom_mean:.2f}\nMean (Gy): {mean:.2f} | SD (Gy): {std:.2f} | argmax(Density) (Gy): {max_density_dose:.2f} | Nominal (Gy): {nominal_dose:.2f}'
        ax.text(1.03, 0.7, annotation_text, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color=color, fontsize=9)

        ax.axvline(x=max_density_dose, color='magenta', linestyle='-', label='Max Density (Gy)')
        ax.axvline(x=mean, color='orange', linestyle='-', label='Mean (Gy)')
        ax.axvline(x=nominal_dose, color='red', linestyle='-', label='Nominal (Gy)')

        # Added loop for plotting dotted gray vertical lines for each quantile
        for quantile_value in [q05_dose, q25_dose, q50_dose, q75_dose, q95_dose]:
            ax.axvline(x=quantile_value, color='gray', linestyle='--', linewidth=1)

        density_color = cmap(norm(binom_mean))

        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 1000)
        y_density = kde(x_grid)

        max_density = np.max(y_density)
        scaling_factor = 1.0 / max_density if max_density > 0 else 1
        scaled_density = y_density * scaling_factor

        ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)

    max_95th_quantile = df_dose_stats_by_voxel[('Dose (Gy)','quantile_95')].max()
    min_5th_quantile = df_dose_stats_by_voxel[('Dose (Gy)','quantile_05')].min()

    # Define legend handles
    legend_handles = [
        Line2D([0], [0], color='magenta', lw=2, linestyle='-', label='Max Density (Gy)'),
        Line2D([0], [0], color='orange', lw=2, linestyle='-', label='Mean (Gy)'),
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='Nominal (Gy)'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Quantiles (05, 25, 50, 75, 95)')
    ]

    for (patient_id, bx_index), group in df_dose.groupby(['Patient ID', 'Bx index']):
        bx_id = group.iloc[0]['Bx ID']

        unique_voxels = group['Voxel index'].unique()
        palette_black = {voxel: "black" for voxel in unique_voxels}


        # Ensure that the FacetGrid is only created for valid voxel indices
        valid_voxel_indices = [voxel for voxel in group['Voxel index'].unique() if voxel in palette_black]
        if len(valid_voxel_indices) == 0:
            print(f"No valid voxel indices for Patient ID: {patient_id}, Bx index: {bx_index}")
            continue
        
        #g = sns.FacetGrid(group[group['Voxel index'].isin(valid_voxel_indices)], row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)
        g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=15, height=1, palette=palette_black)

        g.map(annotate_and_color_v2, "Dose (Gy)")

        g.set(xlim=(min_5th_quantile, max_95th_quantile))

        g.fig.subplots_adjust(right=0.53, left=0.07, top=0.95, bottom=0.05)
        cbar_ax = g.fig.add_axes([0.9, 0.2, 0.03, 0.6])
        g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=False, left=True)
        g.set_axis_labels("Dose (Gy)", "")
        g.fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)
        g.fig.text(0.88, 0.5, 'Tumor tissue score', va='center', rotation='vertical', fontsize=12)

        for ax in g.axes.flat:
            ax.grid(True, which='both', axis='x', linestyle='-', color='gray', linewidth=0.5)
            ax.set_axisbelow(True)

        # Update legend to include quantiles
        g.fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.9, 1), ncol=1, frameon=True, facecolor='white')

        plt.suptitle(f'Patient ID: {patient_id}, Bx ID: {bx_id}', fontsize=16, fontweight='bold', y=0.98)
        
        figure_width_in = svg_image_width / dpi
        figure_height_in = svg_image_height / dpi

        g.fig.set_size_inches(figure_width_in, figure_height_in)

        patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patient_id]
        svg_dose_fig_name = ridge_line_dose_and_binom_general_plot_name_string+' - '+str(patient_id)+' - '+str(bx_id)+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

        plt.close(g.fig)

    df_dose = dataframe_builders.convert_columns_to_categorical_and_downcast(df_dose, threshold=0.25)







def cohort_all_biopsies_dosimtery_boxplot_grouped_by_patient(cohort_global_dosimetry_dataframe,
                                          general_plot_name_string,
                                          cohort_output_figures_dir,
                                          spacing_factor = 0.12):
    import matplotlib.pyplot as plt

    df = cohort_global_dosimetry_dataframe.copy()

    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 8))

    tick_positions = []
    legend_added = False  # Flag to ensure legend is added only once

    for i, row in df.iterrows():
        # Extract values for the current biopsy
        biopsy_values = [
            row["Global q25 dose"],  # Box lower bound
            row["Global q50 dose"],  # Median
            row["Global q75 dose"],  # Box upper bound
            row["Global min dose"],  # Min (point)
            row["Global max dose"],  # Max (point)
            row["Global q05 dose"],  # 5th percentile
            row["Global q95 dose"],  # 95th percentile
            row["Global max density dose"],  # Max density
            row["Global mean dose"],   # Mean
            row["Global nominal mean dose"]  # Global nominal mean
        ]

        # Calculate fences
        iqr = row["Global q75 dose"] - row["Global q25 dose"]
        lower_fence = row["Global q25 dose"] - 1.5 * iqr
        upper_fence = row["Global q75 dose"] + 1.5 * iqr

        # Set x-position
        x_pos = (i + 1) * spacing_factor
        tick_positions.append(x_pos)

        # Plot box-like structure
        ax.hlines(y=biopsy_values[0], xmin=x_pos - 0.05, xmax=x_pos + 0.05, color='blue', lw=2)
        ax.hlines(y=biopsy_values[2], xmin=x_pos - 0.05, xmax=x_pos + 0.05, color='blue', lw=2)
        ax.vlines(x=x_pos, ymin=biopsy_values[0], ymax=biopsy_values[2], color='blue', lw=2)
        ax.plot([x_pos], [biopsy_values[1]], 'ro', label="Median" if not legend_added else "")

        # Add whiskers and fences
        ax.plot([x_pos, x_pos], [lower_fence, biopsy_values[0]], color='black', linestyle='-')  # Lower whisker
        ax.plot([x_pos, x_pos], [biopsy_values[2], upper_fence], color='black', linestyle='-')  # Upper whisker
        ax.plot([x_pos], [lower_fence], 'k_', label="Lower Fence" if not legend_added else "")  # Lower fence
        ax.plot([x_pos], [upper_fence], 'k_', label="Upper Fence" if not legend_added else "")  # Upper fence

        # Add points for other statistics
        ax.scatter([x_pos], [biopsy_values[3]], color='black', label="Min" if not legend_added else "")
        ax.scatter([x_pos], [biopsy_values[4]], color='black', label="Max" if not legend_added else "")
        ax.scatter([x_pos], [biopsy_values[5]], color='green', label="5th Percentile" if not legend_added else "")
        ax.scatter([x_pos], [biopsy_values[6]], color='green', label="95th Percentile" if not legend_added else "")
        ax.scatter([x_pos], [biopsy_values[7]], color='magenta', label="Max Density" if not legend_added else "")
        ax.scatter([x_pos], [biopsy_values[8]], color='orange', label="Mean" if not legend_added else "")
        ax.scatter([x_pos], [biopsy_values[9]], color='red', label="Nominal mean" if not legend_added else "")

        legend_added = True  # Ensure the legend is added only once

    # Customize the plot
    ax.set_title("Box-plot distribution for All Biopsies", fontsize=16)
    ax.set_ylabel("Dose Values", fontsize=12)
    ax.set_xlabel("Biopsy Index", fontsize=12)
    ax.set_xticks(tick_positions)

    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Create combined labels with Patient ID and Bx ID
    x_labels = df["Patient ID"].astype(str) + " - " + df["Bx ID"].astype(str)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=10)

    # Add legend with white background
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", facecolor="white", framealpha=1)

    plt.tight_layout()

    # Save the figure
    svg_boxplot_fig_name = general_plot_name_string + '.svg'
    svg_boxplot_fig_file_path = cohort_output_figures_dir.joinpath(svg_boxplot_fig_name)
    plt.savefig(svg_boxplot_fig_file_path, format='svg', bbox_inches="tight")

    # Close the plot
    plt.close()


























### For the transverse plane accuracy plots cohort

def draw_ellipse(position, covariance, ax, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width = height = 2 * np.sqrt(covariance)
    
    for nsig in range(1, 3):
        ax.add_patch(Ellipse(xy = position, width = nsig * width, height = nsig * height, angle = angle, **kwargs))

def production_plot_transverse_accuracy_with_marginals_and_gaussian_fit(cohort_nearest_dils_dataframe,
                                         general_plot_name_string,
                                        cohort_output_figures_dir):

    plt.ioff()
    df = cohort_nearest_dils_dataframe
    df_filtered = df[(df['Simulated bool'] == False) & (df["Target DIL (by centroids)"] == True)]
    
    """
    # Combine position columns into a single categorical column
    df_filtered['Position Category'] = df_filtered.apply(
        lambda row: f"{row['Bx position in prostate LR']} {row['Bx position in prostate AP']} {row['Bx position in prostate SI']}",
        axis=1
    )

    # Map position categories to colors
    unique_categories = df_filtered['Position Category'].unique()
    color_map = {category: plt.cm.tab20(i / len(unique_categories)) for i, category in enumerate(unique_categories)}

    # Assign colors based on the mapping
    df_filtered['Color'] = df_filtered['Position Category'].map(color_map)
    """

    # Extract values and colors
    x = df_filtered['Bx (X, DIL centroid frame)'].values
    y = df_filtered['Bx (Y, DIL centroid frame)'].values
    #colors = df_filtered['Color'].values

    # Calculate means and standard deviations
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)
    mean = [mean_x, mean_y]
    cov = np.cov(x, y)
    rmse = np.sqrt(mean_squared_error(y, np.full_like(y, mean_y)))  # Simplified RMSE calculation

    # Setup main plot and marginal plots
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 4)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xDist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # Main scatter plot
    ax_main.scatter(x, y, color=colors)
    draw_ellipse(mean, cov, ax=ax_main, alpha=0.2, color='red', edgecolor='black')
    ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Horizontal line
    ax_main.axvline(0, color='gray', linestyle='--', alpha=0.5)  # Vertical line

    # Marginal distributions
    sns.histplot(x=x, ax=ax_xDist, kde=True, stat="density", color="m")
    sns.histplot(y=y, ax=ax_yDist, kde=True, stat="density", color="m")

    # Dotted lines at 0 for marginal distributions
    ax_xDist.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_yDist.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Formatting and cleanup for marginals
    ax_xDist.tick_params(axis="x", labelbottom=False)
    ax_yDist.tick_params(axis="y", labelleft=False)

    # Plot mean as a black cross
    ax_main.plot(mean_x, mean_y, 'kx', markersize=10, markeredgewidth=2)

    # Annotations for mean, std, and RMSE
    ax_main.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax_main.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax_xDist.text(0.95, 0.7, f'Mean: {mean_x:.2f}\nStd: {std_x:.2f}', transform=ax_xDist.transAxes, horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    ax_yDist.text(0.7, 0.95, f'Mean: {mean_y:.2f}\nStd: {std_y:.2f}', transform=ax_yDist.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5), rotation=-90)

    ax_main.set_xlabel('(L/R) (mm)')
    ax_main.set_ylabel('(A/P) (mm)')

    ax_main.text(0.2, 0.02, "Transverse plane", transform=ax_main.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize=12, style='italic')

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)



def production_plot_transverse_accuracy_with_marginals_and_gaussian_fit_global_tissue_score_coloring(cohort_nearest_dils_dataframe,
                                                                                                     cohort_global_tissue_class_dataframe,
                                         general_plot_name_string,
                                        cohort_output_figures_dir):

    plt.ioff()
    df = cohort_nearest_dils_dataframe
    df_filtered = copy.deepcopy(df[(df['Simulated bool'] == False) & (df["Target DIL (by centroids)"] == True)])

    # Rename columns to match df_tissue for a successful merge
    df_filtered.rename(columns={
        'Relative DIL ID': 'Relative structure ROI',
        'Relative struct type': 'Relative structure type',
        'Relative DIL index': 'Relative structure index'
    }, inplace=True)

    df_tissue = cohort_global_tissue_class_dataframe
    
    # Merge df_filtered with df_tissue to get the 'Global mean binom est' for coloring
    df_filtered = df_filtered.merge(
        df_tissue[['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type', 'Global mean binom est']],
        on=['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type'],
        how='left'
    )

    # Define the color mapping based on 'Global mean binom est'
    colors = ["green", "blue", "black"]  # Adjust colors as needed
    cmap = LinearSegmentedColormap.from_list("CustomCmap", colors, N=10)  # More bins for smoother color transitions

    # Normalize the 'Global mean binom est' values for color mapping
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Apply color mapping
    df_filtered['Color'] = df_filtered['Global mean binom est'].apply(lambda x: sm.to_rgba(x))



    # Extract values and colors
    x = df_filtered['Bx (X, DIL centroid frame)'].values
    y = df_filtered['Bx (Y, DIL centroid frame)'].values
    colors = df_filtered['Color'].values

    # Calculate means and standard deviations
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)
    mean = [mean_x, mean_y]
    cov = np.cov(x, y)
    rmse = np.sqrt(mean_squared_error(y, np.full_like(y, mean_y)))  # Simplified RMSE calculation

    # Setup main plot and marginal plots
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(5, 5)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xDist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    cbar_ax = fig.add_axes([0.8, 0.3, 0.05, 0.4])  # Adjust color bar position

    # Main scatter plot
    ax_main.scatter(x, y, c=colors, edgecolor='none', s=100)
    draw_ellipse(mean, cov, ax=ax_main, alpha=0.1, color='red')
    ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Horizontal line
    ax_main.axvline(0, color='gray', linestyle='--', alpha=0.5)  # Vertical line

    # Adding color bar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('DIL specific tumor tissue score')

    # Marginal distributions
    sns.histplot(x=x, ax=ax_xDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})
    sns.histplot(y=y, ax=ax_yDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})

    # Dotted lines at 0 for marginal distributions
    ax_xDist.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_yDist.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Formatting and cleanup for marginals
    ax_xDist.tick_params(axis="x", labelbottom=False)
    ax_yDist.tick_params(axis="y", labelleft=False)

    # Plot mean as a black cross
    ax_main.plot(mean_x, mean_y, 'k+', markersize=10, markeredgewidth=2)
    #plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

    # Annotations for mean, std, and RMSE
    ax_main.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax_main.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax_xDist.text(0, 1.15, r'$\mu$: {:.2f} $\sigma$: {:.2f} (mm)'.format(mean_x, std_x),
              transform=ax_xDist.transAxes, horizontalalignment='left', verticalalignment='top',
              bbox=dict(facecolor='white', alpha=0.5))
    ax_yDist.text(0, 1.05, r'$\mu$: {:.2f} $\sigma$: {:.2f} (mm)'.format(mean_y, std_y),
              transform=ax_yDist.transAxes, horizontalalignment='left', verticalalignment='top',
              bbox=dict(facecolor='white', alpha=0.5))
    #ax_yDist.text(0.7, 0.95, r'Mean: {mean_y:.2f}\nStd: {std_y:.2f}', transform=ax_yDist.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5), rotation=-90)

    ax_main.set_xlabel('(L/R) (mm)')
    ax_main.set_ylabel('(A/P) (mm)')

    ax_main.text(0.2, 0.02, "Transverse plane", transform=ax_main.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize=12, style='italic')

    # Adding point annotations and legend
    unique_patient_ids = df_filtered['Patient ID'].unique()
    patient_id_to_number = {pid: i + 1 for i, pid in enumerate(unique_patient_ids)}

    for i, row in df_filtered.iterrows():
        ax_main.annotate(patient_id_to_number[row['Patient ID']], (row['Bx (X, DIL centroid frame)'], row['Bx (Y, DIL centroid frame)']), fontsize=9, color='red')

    legend_labels = [f"{num}: {pid}" for pid, num in patient_id_to_number.items()]
    legend_text = "\n".join(legend_labels)
    ax_main.text(1.05, 0.5, legend_text, transform=ax_main.transAxes, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))


    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)





def production_plot_transverse_accuracy_with_marginals_and_gaussian_fit_global_tissue_score_coloring_with_table(cohort_nearest_dils_dataframe,
                                                                                                    cohort_global_tissue_class_dataframe,
                                                                                                    general_plot_name_string,
                                                                                                    cohort_output_figures_dir):

    plt.ioff()
    df = cohort_nearest_dils_dataframe
    df_filtered = copy.deepcopy(df[(df['Simulated bool'] == False) & (df["Target DIL (by centroids)"] == True)])

    # Rename columns to match df_tissue for a successful merge
    df_filtered.rename(columns={
        'Relative DIL ID': 'Relative structure ROI',
        'Relative struct type': 'Relative structure type',
        'Relative DIL index': 'Relative structure index'
    }, inplace=True)

    df_tissue = cohort_global_tissue_class_dataframe

    # Merge df_filtered with df_tissue to get the 'Global mean binom est' for coloring
    df_filtered = df_filtered.merge(
        df_tissue[['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type', 'Global mean binom est']],
        on=['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type'],
        how='left'
    )

    # Define the color mapping based on 'Global mean binom est'
    colors = ["green", "blue", "black"]  # Adjust colors as needed
    cmap = LinearSegmentedColormap.from_list("CustomCmap", colors, N=10)  # More bins for smoother color transitions

    # Normalize the 'Global mean binom est' values for color mapping
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Apply color mapping
    df_filtered['Color'] = df_filtered['Global mean binom est'].apply(lambda x: sm.to_rgba(x))

    # Extract values and colors
    x = df_filtered['Bx (X, DIL centroid frame)'].values
    y = df_filtered['Bx (Y, DIL centroid frame)'].values
    colors = df_filtered['Color'].values

    # Calculate means and standard deviations
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)
    mean = [mean_x, mean_y]
    cov = np.cov(x, y)
    rmse = np.sqrt(mean_squared_error(y, np.full_like(y, mean_y)))  # Simplified RMSE calculation

    # Setup main plot and marginal plots
    fig = plt.figure(figsize=(24, 12))
    ax_main = fig.add_subplot(111)

    # Main scatter plot
    scatter = ax_main.scatter(x, y, c=colors, edgecolor='none', s=100)
    draw_ellipse(mean, cov, ax=ax_main, alpha=0.1, color='red')
    ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Horizontal line
    ax_main.axvline(0, color='gray', linestyle='--', alpha=0.5)  # Vertical line

    # Ensure the main plot is square with equal increments on both axes
    ax_main.set_aspect('equal', adjustable='box')

    # Determine the symmetric limits around zero
    max_limit = round(max(abs(x).max(), abs(y).max())) + max(1, round(0.05 * round(max(abs(x).max(), abs(y).max()))))  # the addition adds 5% of padding of the max limit

    # Set symmetric limits and ticks around zero
    ax_main.set_xlim([-max_limit, max_limit])
    ax_main.set_ylim([max_limit, -max_limit])  # Flip the y-axis

    # Set tick increments
    tick_increment = 1  # Change this value to your desired tick increment
    ax_main.set_xticks(np.arange(-max_limit, max_limit + tick_increment, tick_increment))
    ax_main.set_yticks(np.arange(-max_limit, max_limit + tick_increment, tick_increment))

    # Use make_axes_locatable to create new axes that are dynamically sized relative to the main plot
    divider = make_axes_locatable(ax_main)
    ax_xDist = divider.append_axes("top", size="15%", pad=0.5, sharex=ax_main)
    ax_yDist = divider.append_axes("right", size="15%", pad=0.5, sharey=ax_main)
    ax_cbar = divider.append_axes("left", size="5%", pad=2)
    ax_legend = divider.append_axes("right", size="10%", pad=1)
    ax_table = divider.append_axes("right", size="175%", pad=1)

    # Adding color bar
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.set_label('DIL specific tumor tissue score')

    # Marginal distributions
    sns.histplot(x=x, ax=ax_xDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})
    sns.histplot(y=y, ax=ax_yDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})

    # Dotted lines at 0 for marginal distributions
    ax_xDist.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_yDist.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Formatting and cleanup for marginals
    ax_xDist.tick_params(axis="x", labelbottom=False)
    ax_yDist.tick_params(axis="y", labelleft=False)

    # Plot mean as a black cross
    ax_main.plot(mean_x, mean_y, 'k+', markersize=10, markeredgewidth=2)

    # Annotations for mean, std, and RMSE
    mean_std_text = f'$\mu_x$: {mean_x:.2f} $\sigma_x$: {std_x:.2f} (mm)\n$\mu_y$: {mean_y:.2f} $\sigma_y$: {std_y:.2f} (mm)'
    fontsize = 10
    text_area = TextArea(mean_std_text, textprops=dict(fontsize=fontsize))
    annotation_box = AnnotationBbox(text_area, (0.5, 1.1), xycoords='axes fraction',
                                    boxcoords="axes fraction", box_alignment=(0.5, 0), frameon=True,
                                    bboxprops=dict(facecolor='white', edgecolor='black'))
    ax_xDist.add_artist(annotation_box)

    # Adjust text size dynamically
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_annotation = text_area.get_window_extent(renderer)
    while bbox_annotation.width > ax_xDist.bbox.width or bbox_annotation.height > ax_xDist.bbox.height:
        fontsize -= 1
        text_area = TextArea(mean_std_text, textprops=dict(fontsize=fontsize))
        annotation_box = AnnotationBbox(text_area, (0.5, 1.1), xycoords='axes fraction',
                                        boxcoords="axes fraction", box_alignment=(0.5, 0), frameon=True,
                                        bboxprops=dict(facecolor='white', edgecolor='black'))
        ax_xDist.add_artist(annotation_box)
        fig.canvas.draw()
        bbox_annotation = text_area.get_window_extent(renderer)

    ax_main.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax_main.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Add direction annotations outside the plot area
    ax_main.annotate('L', xy=(1, 0), xytext=(0, -30),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='right', va='center', fontsize=12)
    ax_main.annotate('R', xy=(0, 0), xytext=(0, -30),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=12)
    ax_main.annotate('A', xy=(0, 1), xytext=(-40, 0),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=12)
    ax_main.annotate('P', xy=(0, 0), xytext=(-40, 0),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=12)

    ax_main.set_xlabel('(L/R) (mm)')
    ax_main.set_ylabel('(A/P) (mm)')

    ax_main.text(0.2, 0.02, "Transverse plane", transform=ax_main.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize=12, style='italic')

    # Adding point annotations and legend
    unique_patient_ids = df_filtered['Patient ID'].unique()
    patient_id_to_number = {pid: i + 1 for i, pid in enumerate(unique_patient_ids)}

    unique_bx_ids = df_filtered.index
    bx_id_to_letter = {index: letter for index, letter in zip(unique_bx_ids, string.ascii_lowercase)}

    annotation_strings = []
    table_data = []

    for i, row in df_filtered.iterrows():
        annotation_str = f"{patient_id_to_number[row['Patient ID']]}-{bx_id_to_letter[i]}"
        annotation_strings.append(annotation_str)
        table_data.append([
            row['Patient ID'],  # Add Patient ID column
            annotation_str,
            row['Bx ID'],
            f"{row['Global mean binom est']:.2f}",  # Ensure two decimal places
            f"{row['BX to DIL centroid distance']:.2f}",
            row['Bx position in prostate LR'],
            row['Bx position in prostate AP'],
            row['Bx position in prostate SI']
        ])
        ax_main.annotate(annotation_str, 
                        (row['Bx (X, DIL centroid frame)'], row['Bx (Y, DIL centroid frame)']),
                        fontsize=9, color='black', 
                        textcoords="offset points", xytext=(5,5))  # Offset by (5, 5)

    # Create a DataFrame for the table data
    df_table = pandas.DataFrame(table_data, columns=["Patient ID", "Annotation", "Bx ID", "DIL sp. TTS*", 'Bx Cent. to DIL Cent. (mm)', "Bx position LR", "Bx position AP", "Bx position SI"])

    # Create the table with the "Patient ID" column
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=df_table.values, 
                        colLabels=df_table.columns, 
                        cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Adjust column widths to minimize white space
    max_colwidths = [max([len(str(cell)) for cell in df_table[col]]) for col in df_table.columns]
    table.auto_set_column_width(list(range(len(df_table.columns))))
    for col_idx, col in enumerate(df_table.columns):
        for cell in table.get_celld().values():
            if cell.get_text().get_text() == col:
                cell.set_width(max_colwidths[col_idx] * 0.1)  # Adjust the multiplier as needed

    # Adjust row heights to minimize squishing
    # for row_idx in range(len(df_table)):
    #     max_row_height = max([len(str(cell)) for cell in df_table.iloc[row_idx]])
    #     table.scale(1, max_row_height * 0.1)  # Adjust the multiplier as needed

    # Draw the figure to update the table position
    fig.canvas.draw()

    # Calculate the position below the table using the updated bbox
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds

    # Add italicized annotation below the table
    fig.text(x0 + width / 2, y0 - 0.02, "*TTS = Tumor tissue score", ha='center', va='top', fontsize=10, style='italic', transform=fig.transFigure)

    # Create the legend text
    legend_labels = [f"{num}: {pid}" for pid, num in patient_id_to_number.items()]
    legend_text = "\n".join(legend_labels)
    ax_legend.text(0.5, 0.5, legend_text, transform=ax_legend.transAxes, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    ax_legend.axis('off')

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)





def production_plot_sagittal_accuracy_with_marginals_and_gaussian_fit_global_tissue_score_coloring_with_table(cohort_nearest_dils_dataframe,
                                                                                                    cohort_global_tissue_class_dataframe,
                                                                                                    general_plot_name_string,
                                                                                                    cohort_output_figures_dir):
    
    plt.ioff()
    df = cohort_nearest_dils_dataframe
    df_filtered = copy.deepcopy(df[(df['Simulated bool'] == False) & (df["Target DIL (by centroids)"] == True)])

    # Rename columns to match df_tissue for a successful merge
    df_filtered.rename(columns={
        'Relative DIL ID': 'Relative structure ROI',
        'Relative struct type': 'Relative structure type',
        'Relative DIL index': 'Relative structure index'
    }, inplace=True)

    df_tissue = cohort_global_tissue_class_dataframe

    # Merge df_filtered with df_tissue to get the 'Global mean binom est' for coloring
    df_filtered = df_filtered.merge(
        df_tissue[['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type', 'Global mean binom est']],
        on=['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type'],
        how='left'
    )

    # Define the color mapping based on 'Global mean binom est'
    colors = ["green", "blue", "black"]  # Adjust colors as needed
    cmap = LinearSegmentedColormap.from_list("CustomCmap", colors, N=10)  # More bins for smoother color transitions

    # Normalize the 'Global mean binom est' values for color mapping
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Apply color mapping
    df_filtered['Color'] = df_filtered['Global mean binom est'].apply(lambda x: sm.to_rgba(x))

    # Extract values and colors
    x = df_filtered['Bx (Z, DIL centroid frame)'].values
    y = df_filtered['Bx (Y, DIL centroid frame)'].values
    colors = df_filtered['Color'].values

    # Calculate means and standard deviations
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)
    mean = [mean_x, mean_y]
    cov = np.cov(x, y)
    rmse = np.sqrt(mean_squared_error(y, np.full_like(y, mean_y)))  # Simplified RMSE calculation

    # Setup main plot and marginal plots
    fig = plt.figure(figsize=(24, 12))
    ax_main = fig.add_subplot(111)

    # Main scatter plot
    scatter = ax_main.scatter(x, y, c=colors, edgecolor='none', s=100)
    draw_ellipse(mean, cov, ax=ax_main, alpha=0.1, color='red')
    ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Horizontal line
    ax_main.axvline(0, color='gray', linestyle='--', alpha=0.5)  # Vertical line

    # Ensure the main plot is square with equal increments on both axes
    ax_main.set_aspect('equal', adjustable='box')

    # Determine the symmetric limits around zero
    max_limit = round(max(abs(x).max(), abs(y).max())) + max(1, round(0.05 * round(max(abs(x).max(), abs(y).max()))))  # the addition adds 5% of padding of the max limit

    # Set symmetric limits and ticks around zero
    ax_main.set_xlim([-max_limit, max_limit])
    ax_main.set_ylim([max_limit, -max_limit])  # Flip the y-axis

    # Set tick increments
    tick_increment = 1  # Change this value to your desired tick increment
    ax_main.set_xticks(np.arange(-max_limit, max_limit + tick_increment, tick_increment))
    ax_main.set_yticks(np.arange(-max_limit, max_limit + tick_increment, tick_increment))

    # Use make_axes_locatable to create new axes that are dynamically sized relative to the main plot
    divider = make_axes_locatable(ax_main)
    ax_xDist = divider.append_axes("top", size="15%", pad=0.5, sharex=ax_main)
    ax_yDist = divider.append_axes("right", size="15%", pad=0.5, sharey=ax_main)
    ax_cbar = divider.append_axes("left", size="5%", pad=2)
    ax_legend = divider.append_axes("right", size="10%", pad=1)
    ax_table = divider.append_axes("right", size="175%", pad=1)

    # Adding color bar
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.set_label('DIL specific tumor tissue score')

    # Marginal distributions
    sns.histplot(x=x, ax=ax_xDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})
    sns.histplot(y=y, ax=ax_yDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})

    # Dotted lines at 0 for marginal distributions
    ax_xDist.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_yDist.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Formatting and cleanup for marginals
    ax_xDist.tick_params(axis="x", labelbottom=False)
    ax_yDist.tick_params(axis="y", labelleft=False)

    # Plot mean as a black cross
    ax_main.plot(mean_x, mean_y, 'k+', markersize=10, markeredgewidth=2)

    # Annotations for mean, std, and RMSE
    mean_std_text = f'$\mu_z$: {mean_x:.2f} $\sigma_z$: {std_x:.2f} (mm)\n$\mu_y$: {mean_y:.2f} $\sigma_y$: {std_y:.2f} (mm)'
    fontsize = 10
    text_area = TextArea(mean_std_text, textprops=dict(fontsize=fontsize))
    annotation_box = AnnotationBbox(text_area, (0.5, 1.1), xycoords='axes fraction',
                                    boxcoords="axes fraction", box_alignment=(0.5, 0), frameon=True,
                                    bboxprops=dict(facecolor='white', edgecolor='black'))
    ax_xDist.add_artist(annotation_box)

    # Adjust text size dynamically
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_annotation = text_area.get_window_extent(renderer)
    while bbox_annotation.width > ax_xDist.bbox.width or bbox_annotation.height > ax_xDist.bbox.height:
        fontsize -= 1
        text_area = TextArea(mean_std_text, textprops=dict(fontsize=fontsize))
        annotation_box = AnnotationBbox(text_area, (0.5, 1.1), xycoords='axes fraction',
                                        boxcoords="axes fraction", box_alignment=(0.5, 0), frameon=True,
                                        bboxprops=dict(facecolor='white', edgecolor='black'))
        ax_xDist.add_artist(annotation_box)
        fig.canvas.draw()
        bbox_annotation = text_area.get_window_extent(renderer)

    ax_main.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax_main.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Add direction annotations outside the plot area
    ax_main.annotate('S', xy=(1, 0), xytext=(0, -30),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='right', va='center', fontsize=12)
    ax_main.annotate('I', xy=(0, 0), xytext=(0, -30),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=12)
    ax_main.annotate('A', xy=(0, 1), xytext=(-40, 0),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=12)
    ax_main.annotate('P', xy=(0, 0), xytext=(-40, 0),
                    xycoords=('axes fraction', 'axes fraction'),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=12)

    ax_main.set_xlabel('(S/I) (mm)')
    ax_main.set_ylabel('(A/P) (mm)')

    ax_main.text(0.2, 0.02, "Sagittal plane", transform=ax_main.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize=12, style='italic')

    # Adding point annotations and legend
    unique_patient_ids = df_filtered['Patient ID'].unique()
    patient_id_to_number = {pid: i + 1 for i, pid in enumerate(unique_patient_ids)}

    unique_bx_ids = df_filtered.index
    bx_id_to_letter = {index: letter for index, letter in zip(unique_bx_ids, string.ascii_lowercase)}

    annotation_strings = []
    table_data = []

    for i, row in df_filtered.iterrows():
        annotation_str = f"{patient_id_to_number[row['Patient ID']]}-{bx_id_to_letter[i]}"
        annotation_strings.append(annotation_str)
        table_data.append([
            row['Patient ID'],  # Add Patient ID column
            annotation_str,
            row['Bx ID'],
            f"{row['Global mean binom est']:.2f}",  # Ensure two decimal places
            f"{row['BX to DIL centroid distance']:.2f}",
            row['Bx position in prostate LR'],
            row['Bx position in prostate AP'],
            row['Bx position in prostate SI']
        ])
        ax_main.annotate(annotation_str, 
                        (row['Bx (Z, DIL centroid frame)'], row['Bx (Y, DIL centroid frame)']),
                        fontsize=9, color='black', 
                        textcoords="offset points", xytext=(5,5))  # Offset by (5, 5)

    # Create a DataFrame for the table data
    df_table = pandas.DataFrame(table_data, columns=["Patient ID", "Annotation", "Bx ID", "DIL sp. TTS*", 'Bx Cent. to DIL Cent. (mm)', "Bx position LR", "Bx position AP", "Bx position SI"])

    # Create the table with the "Patient ID" column
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=df_table.values, 
                        colLabels=df_table.columns, 
                        cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Adjust column widths to minimize white space
    max_colwidths = [max([len(str(cell)) for cell in df_table[col]]) for col in df_table.columns]
    table.auto_set_column_width(list(range(len(df_table.columns))))
    for col_idx, col in enumerate(df_table.columns):
        for cell in table.get_celld().values():
            if cell.get_text().get_text() == col:
                cell.set_width(max_colwidths[col_idx] * 0.1)  # Adjust the multiplier as needed

    # Adjust row heights to minimize squishing
    # for row_idx in range(len(df_table)):
    #     max_row_height = max([len(str(cell)) for cell in df_table.iloc[row_idx]])
    #     table.scale(1, max_row_height * 0.1)  # Adjust the multiplier as needed

    # Draw the figure to update the table position
    fig.canvas.draw()

    # Calculate the position below the table using the updated bbox
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds

    # Add italicized annotation below the table
    fig.text(x0 + width / 2, y0 - 0.02, "*TTS = Tumor tissue score", ha='center', va='top', fontsize=10, style='italic', transform=fig.transFigure)

    # Create the legend text
    legend_labels = [f"{num}: {pid}" for pid, num in patient_id_to_number.items()]
    legend_text = "\n".join(legend_labels)
    ax_legend.text(0.5, 0.5, legend_text, transform=ax_legend.transAxes, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
    ax_legend.axis('off')

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)





def production_plot_sagittal_accuracy_with_marginals_and_gaussian_fit(cohort_nearest_dils_dataframe,
                                         general_plot_name_string,
                                        cohort_output_figures_dir):
    plt.ioff()
    df = cohort_nearest_dils_dataframe
    df_filtered = df[df['Simulated bool'] == False]
    x = df_filtered['Bx (Z, DIL centroid frame)'].values
    y = df_filtered['Bx (Y, DIL centroid frame)'].values

    # Calculate means and standard deviations
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)
    mean = [mean_x, mean_y]
    cov = np.cov(x, y)
    rmse = np.sqrt(mean_squared_error(y, np.full_like(y, mean_y)))  # Simplified RMSE calculation

    # Setup main plot and marginal plots
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 4)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xDist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # Main scatter plot
    ax_main.scatter(x, y, alpha=0.5)
    draw_ellipse(mean, cov, ax=ax_main, alpha=0.2, color='red', edgecolor='black')
    ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Horizontal line
    ax_main.axvline(0, color='gray', linestyle='--', alpha=0.5)  # Vertical line

    # Marginal distributions
    sns.histplot(x=x, ax=ax_xDist, kde=True, stat="density", color="m")
    sns.histplot(y=y, ax=ax_yDist, kde=True, stat="density", color="m")

    # Dotted lines at 0 for marginal distributions
    ax_xDist.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_yDist.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Formatting and cleanup for marginals
    ax_xDist.tick_params(axis="x", labelbottom=False)
    ax_yDist.tick_params(axis="y", labelleft=False)

    # Plot mean as a black cross
    ax_main.plot(mean_x, mean_y, 'kx', markersize=10, markeredgewidth=2)

    # Annotations for mean, std, and RMSE
    ax_main.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax_main.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax_xDist.text(0.95, 0.7, f'Mean: {mean_x:.2f}\nStd: {std_x:.2f}', transform=ax_xDist.transAxes, horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    ax_yDist.text(0.7, 0.95, f'Mean: {mean_y:.2f}\nStd: {std_y:.2f}', transform=ax_yDist.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5), rotation=-90)

    ax_main.set_xlabel('(S/I) (mm)')
    ax_main.set_ylabel('(A/P) (mm)')

    ax_main.text(0.2, 0.02, "Sagittal plane", transform=ax_main.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize=12, style='italic')

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')



def production_plot_sagittal_accuracy_with_marginals_and_gaussian_fit_global_tissue_score_coloring(cohort_nearest_dils_dataframe,
                                                                                                    cohort_global_tissue_class_dataframe,
                                                                                                    general_plot_name_string,
                                                                                                    cohort_output_figures_dir):

    plt.ioff()
    df = cohort_nearest_dils_dataframe
    df_filtered = copy.deepcopy(df[(df['Simulated bool'] == False) & (df["Target DIL (by centroids)"] == True)])

    # Rename columns to match df_tissue for a successful merge
    df_filtered.rename(columns={
        'Relative DIL ID': 'Relative structure ROI',
        'Relative struct type': 'Relative structure type',
        'Relative DIL index': 'Relative structure index'
    }, inplace=True)

    df_tissue = cohort_global_tissue_class_dataframe
    
    # Merge df_filtered with df_tissue to get the 'Global mean binom est' for coloring
    df_filtered = df_filtered.merge(
        df_tissue[['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type', 'Global mean binom est']],
        on=['Patient ID', 'Bx ID', 'Bx refnum', 'Bx index', 'Relative structure index', 'Relative structure type'],
        how='left'
    )

    # Define the color mapping based on 'Global mean binom est'
    colors = ["green", "blue", "black"]  # Adjust colors as needed
    cmap = LinearSegmentedColormap.from_list("CustomCmap", colors, N=10)  # More bins for smoother color transitions

    # Normalize the 'Global mean binom est' values for color mapping
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Apply color mapping
    df_filtered['Color'] = df_filtered['Global mean binom est'].apply(lambda x: sm.to_rgba(x))



    # Extract values and colors
    x = df_filtered['Bx (Z, DIL centroid frame)'].values
    y = df_filtered['Bx (Y, DIL centroid frame)'].values
    colors = df_filtered['Color'].values

    # Calculate means and standard deviations
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)
    mean = [mean_x, mean_y]
    cov = np.cov(x, y)
    rmse = np.sqrt(mean_squared_error(y, np.full_like(y, mean_y)))  # Simplified RMSE calculation

    # Setup main plot and marginal plots
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(5, 5)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xDist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    cbar_ax = fig.add_axes([0.8, 0.3, 0.05, 0.4])  # Adjust color bar position

    # Main scatter plot
    ax_main.scatter(x, y, c=colors, edgecolor='none', s=100)
    draw_ellipse(mean, cov, ax=ax_main, alpha=0.1, color='red')
    ax_main.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Horizontal line
    ax_main.axvline(0, color='gray', linestyle='--', alpha=0.5)  # Vertical line

    # Adding color bar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('DIL specific tumor tissue score')

    # Marginal distributions
    sns.histplot(x=x, ax=ax_xDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})
    sns.histplot(y=y, ax=ax_yDist, kde=True, stat="density", color='skyblue', line_kws={'color': 'k', 'lw': 2, 'linestyle': '-'})

    # Dotted lines at 0 for marginal distributions
    ax_xDist.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_yDist.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Formatting and cleanup for marginals
    ax_xDist.tick_params(axis="x", labelbottom=False)
    ax_yDist.tick_params(axis="y", labelleft=False)

    # Plot mean as a black cross
    ax_main.plot(mean_x, mean_y, 'k+', markersize=10, markeredgewidth=2)
    #plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

    # Annotations for mean, std, and RMSE
    ax_main.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax_main.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax_xDist.text(0, 1.15, r'$\mu$: {:.2f} $\sigma$: {:.2f} (mm)'.format(mean_x, std_x),
              transform=ax_xDist.transAxes, horizontalalignment='left', verticalalignment='top',
              bbox=dict(facecolor='white', alpha=0.5))
    ax_yDist.text(0, 1.05, r'$\mu$: {:.2f} $\sigma$: {:.2f} (mm)'.format(mean_y, std_y),
              transform=ax_yDist.transAxes, horizontalalignment='left', verticalalignment='top',
              bbox=dict(facecolor='white', alpha=0.5))
    #ax_yDist.text(0.7, 0.95, r'Mean: {mean_y:.2f}\nStd: {std_y:.2f}', transform=ax_yDist.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5), rotation=-90)

    ax_main.set_xlabel('(S/I) (mm)')
    ax_main.set_ylabel('(A/P) (mm)')

    ax_main.text(0.2, 0.02, "Sagittal plane", transform=ax_main.transAxes, horizontalalignment='center', verticalalignment='bottom', fontsize=12, style='italic')

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)



def production_plot_cohort_scatter_plot_matrix_bx_centroids_real_in_dil_frame(cohort_nearest_dils_dataframe,
                                                                              general_plot_name_string,
                                                                              cohort_output_figures_dir):
    plt.ioff()
    df = cohort_nearest_dils_dataframe
    # Filter out rows where 'Simulated bool' is not False
    #df_filtered = df[df['Simulated bool'] == False]
    df_filtered = copy.deepcopy(df[(df['Simulated bool'] == False) & (df["Target DIL (by centroids)"] == True)])

    
    # Create a copy of df_filtered to rename without altering the original DataFrame
    df_renamed = df_filtered.copy()
    # Renaming the columns for plotting
    rename_dict = {
        'Bx (X, DIL centroid frame)': '(L/R)',
        'Bx (Y, DIL centroid frame)': '(A/P)',
        'Bx (Z, DIL centroid frame)': '(S/I)',
    }
    df_renamed.rename(columns=rename_dict, inplace=True)
    
    # Subset the DataFrame with renamed columns for plotting
    cols_to_plot = list(rename_dict.values())  # ['(L/R)', '(A/P)', '(S/I)']
    
    # Pairplot with Seaborn with renamed columns
    g = sns.pairplot(df_renamed[cols_to_plot], diag_kind='kde',
                     plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'},
                     diag_kws={'shade':True})
    
    # Adjusting scatter plots to center at (0, 0) and annotating the diagonal
    for i in range(len(cols_to_plot)):
        for j in range(len(cols_to_plot)):
            ax = g.axes[i][j]
            if i != j:  # For scatter plots
                ax.axhline(0, ls='--', c='gray', lw=1)
                ax.axvline(0, ls='--', c='gray', lw=1)
            else:  # For the diagonal (distribution plots)
                mean_val = df_renamed[cols_to_plot[i]].mean()
                std_val = df_renamed[cols_to_plot[i]].std()
                # Place text for mean and std dev
                ax.text(0.95, 0.9, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}',
                        verticalalignment='top', horizontalalignment='right',
                        transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Update plot labels with the new variable names
    for i in range(len(cols_to_plot)):
        for j in range(len(cols_to_plot)):
            # Set x-axis label
            g.axes[len(cols_to_plot)-1][j].set_xlabel(cols_to_plot[j], fontsize=12)
            # Set y-axis label
            g.axes[i][0].set_ylabel(cols_to_plot[i], fontsize=12)
    
    plt.tight_layout()


    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')



def production_plot_cohort_scatter_plot_matrix_bx_centroids_real_in_dil_frame_v2(cohort_nearest_dils_dataframe,
                                                                              general_plot_name_string,
                                                                              cohort_output_figures_dir):


    plt.ioff()
    df = cohort_nearest_dils_dataframe
    df_filtered = copy.deepcopy(df[(df['Simulated bool'] == False) & (df["Target DIL (by centroids)"] == True)])

    # Rename columns for plotting
    rename_dict = {
        'Bx (X, DIL centroid frame)': '(L/R)',
        'Bx (Y, DIL centroid frame)': '(A/P)',
        'Bx (Z, DIL centroid frame)': '(S/I)',
    }
    df_filtered.rename(columns=rename_dict, inplace=True)

    # Define columns to plot
    cols_to_plot = list(rename_dict.values())  # ['(L/R)', '(A/P)', '(S/I)']

    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=len(cols_to_plot), figsize=(15, 15))

    # Calculate universal limits for all plots with padding
    max_limit = max(df_filtered[cols_to_plot].max().max(), abs(df_filtered[cols_to_plot].min().min()))
    max_limit *= 1.1  # Add 10% padding

    # Define dictionary for annotations on scatter axes
    annotation_dict_map = {'(L/R)+': 'L',
                       '(L/R)-': 'R',
                       '(S/I)+': 'S',
                       '(S/I)-': 'I',
                       '(A/P)-': 'A',
                       '(A/P)+': 'P'}

    # Loop through each subplot position in the grid
    for i, row_var in enumerate(cols_to_plot):
        for j, col_var in enumerate(cols_to_plot):
            ax = axes[i][j]
            if i == j:  # Diagonal: KDE plot
                data = df_filtered[col_var]
                kde = gaussian_kde(data)
                x = np.linspace(-max_limit, max_limit, 500)
                density = kde(x)
                ax.plot(x, density, color='blue', alpha=0.6)  # Plot the KDE line
                ax.fill_between(x, density, color='blue', alpha=0.3)  # Shade under the curve
                ax.set_xlim(-max_limit, max_limit)
                ax.set_ylim(0, max(density))  # Adjust y-axis limits to match the density
                ax.set_ylabel("Density")  # Set y-axis label for KDE plots
                ax.axvline(0, ls='--', color='gray', lw=1)  # Only vertical line at x=0
                # Display mean and std dev
                mean_val = np.mean(data)
                std_val = np.std(data)
                textstr = f'Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}'
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            else:  # Off-diagonal: Scatter plot
                sns.scatterplot(x=df_filtered[col_var], y=df_filtered[row_var], ax=ax, alpha=0.6, edgecolor='k', s=40)
                ax.set_xlim(-max_limit, max_limit)
                ax.set_ylim(-max_limit, max_limit)
                ax.axhline(0, ls='--', color='gray', lw=1)  # Horizontal line at y=0
                ax.axvline(0, ls='--', color='gray', lw=1)  # Vertical line at x=0

            # Set labels and annotations specific to their positions
            if i == len(cols_to_plot) - 1:
                ax.set_xlabel(col_var)
            if j == 0:  # Set y-label only for the left column
                ax.set_ylabel(row_var)

            # Annotation positioning
            if j == 0:
                ax.annotate(annotation_dict_map[row_var + '-'], xy=(0, 0), xycoords='axes fraction',
                            xytext=(-40, 0), textcoords='offset points',
                            horizontalalignment='right', verticalalignment='center', fontsize=12)
                ax.annotate(annotation_dict_map[row_var + '+'], xy=(0, 1), xycoords='axes fraction',
                            xytext=(-40, 0), textcoords='offset points',
                            horizontalalignment='right', verticalalignment='center', fontsize=12)

            if i == len(cols_to_plot) - 1:
                ax.annotate(annotation_dict_map[col_var + '-'], xy=(0, 0), xycoords='axes fraction',
                            xytext=(0, -20), textcoords='offset points',
                            horizontalalignment='center', verticalalignment='top', fontsize=12)
                ax.annotate(annotation_dict_map[col_var + '+'], xy=(1, 0), xycoords='axes fraction',
                            xytext=(0, -20), textcoords='offset points',
                            horizontalalignment='center', verticalalignment='top', fontsize=12)

    plt.tight_layout()

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)







def production_plot_cohort_double_sextant_biopsy_distribution(cohort_biopsy_basic_spatial_features_dataframe,
                                                              general_plot_name_string,
                                                              cohort_output_figures_dir
                                                              ):

    df = cohort_biopsy_basic_spatial_features_dataframe
    """
    # Sample DataFrame setup (replace this with your actual DataFrame)
    data = {
        'Patient ID': [1, 1, 2, 2, 3],
        'Bx ID': [101, 102, 103, 104, 105],
        'Simulated bool': [False, True, False, True, False],
        'Simulated type': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1'],
        'Struct type': ['TypeA', 'TypeB', 'TypeA', 'TypeB', 'TypeA'],
        'Bx refnum': [1, 1, 2, 2, 3],
        'Bx index': [5, 4, 6, 7, 3],
        'Bx position in prostate LR': ['Left', 'Right', 'Left', 'Right', 'Left'],
        'Bx position in prostate AP': ['Anterior', 'Posterior', 'Anterior', 'Posterior', 'Anterior'],
        'Bx position in prostate SI': ['Base', 'Apex', 'Mid', 'Base', 'Mid']
    }

    df = pd.DataFrame(data)
    """

    # Define the double sextant mapping
    # Create sextant keys based on the LR, AP, and SI positions
    def create_sextant_key(row):
        return f"{row['Bx position in prostate LR']}-{row['Bx position in prostate SI']}"

    df['Sextant'] = df.apply(create_sextant_key, axis=1)

    # Count occurrences of biopsies in each sextant for anterior and posterior slices
    anterior_df = df[df['Bx position in prostate AP'] == 'Anterior']
    posterior_df = df[df['Bx position in prostate AP'] == 'Posterior']

    anterior_counts = anterior_df['Sextant'].value_counts().to_dict()
    posterior_counts = posterior_df['Sextant'].value_counts().to_dict()

    # Setup sextant regions and fill in with biopsy counts
    sextant_regions = ['Left-Base (Superior)', 'Right-Base (Superior)', 
                   'Left-Mid', 'Right-Mid', 
                   'Left-Apex (Inferior)', 'Right-Apex (Inferior)']


    anterior_data = [anterior_counts.get(sextant, 0) for sextant in sextant_regions]
    posterior_data = [posterior_counts.get(sextant, 0) for sextant in sextant_regions]

    # Convert the data into a 2x3 grid for each slice (anterior and posterior)
    # Base is now on the top, and Apex on the bottom
    anterior_matrix = np.array([
        [anterior_data[0], anterior_data[1]],  # Base row
        [anterior_data[2], anterior_data[3]],  # Mid row
        [anterior_data[4], anterior_data[5]]   # Apex row
    ])

    posterior_matrix = np.array([
        [posterior_data[0], posterior_data[1]],  # Base row
        [posterior_data[2], posterior_data[3]],  # Mid row
        [posterior_data[4], posterior_data[5]]   # Apex row
    ])

    # Plot the heatmaps for anterior and posterior slices
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Anterior heatmap
    sns.heatmap(anterior_matrix, annot=True, cmap="Reds", fmt="d", cbar_kws={'label': 'Number of Biopsies'}, ax=axs[0])
    axs[0].set_title('Anterior Slice')
    axs[0].set_xticklabels(['Left', 'Right'])
    axs[0].set_yticklabels(['Base', 'Mid', 'Apex'])

    # Posterior heatmap
    sns.heatmap(posterior_matrix, annot=True, cmap="Blues", fmt="d", cbar_kws={'label': 'Number of Biopsies'}, ax=axs[1])
    axs[1].set_title('Posterior Slice')
    axs[1].set_xticklabels(['Left', 'Right'])
    axs[1].set_yticklabels(['Base', 'Mid', 'Apex'])

    plt.tight_layout()

    svg_dose_fig_name = general_plot_name_string+'.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)




def production_plot_cohort_double_sextant_dil_distribution(cohort_dil_spatial_features_dataframe,
                                                           general_plot_name_string,
                                                           cohort_output_figures_dir):
    df = cohort_dil_spatial_features_dataframe
    
    # Filter for 'DIL ref' structures and Patient IDs containing 'F2'
    df = df[(df['Structure type'] == 'DIL ref') & (df['Patient ID'].str.contains('F2'))]
    
    # Define the double sextant mapping based on the DIL location
    def create_sextant_key(row):
        return f"{row['DIL prostate sextant (LR)']}-{row['DIL prostate sextant (SI)']}"
    
    df['Sextant'] = df.apply(create_sextant_key, axis=1)
    
    # Count occurrences of DILs in each sextant for anterior and posterior slices
    anterior_df = df[df['DIL prostate sextant (AP)'] == 'Anterior']
    posterior_df = df[df['DIL prostate sextant (AP)'] == 'Posterior']

    anterior_counts = anterior_df['Sextant'].value_counts().to_dict()
    posterior_counts = posterior_df['Sextant'].value_counts().to_dict()
    
    # Define sextant regions
    sextant_regions = ['Left-Base (Superior)', 'Right-Base (Superior)', 
                       'Left-Mid', 'Right-Mid', 
                       'Left-Apex (Inferior)', 'Right-Apex (Inferior)']

    anterior_data = [anterior_counts.get(sextant, 0) for sextant in sextant_regions]
    posterior_data = [posterior_counts.get(sextant, 0) for sextant in sextant_regions]

    # Convert data to 2x3 grids for heatmaps
    anterior_matrix = np.array([
        [anterior_data[0], anterior_data[1]],  # Base row
        [anterior_data[2], anterior_data[3]],  # Mid row
        [anterior_data[4], anterior_data[5]]   # Apex row
    ])

    posterior_matrix = np.array([
        [posterior_data[0], posterior_data[1]],  # Base row
        [posterior_data[2], posterior_data[3]],  # Mid row
        [posterior_data[4], posterior_data[5]]   # Apex row
    ])

    # Plot heatmaps for anterior and posterior slices
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Anterior heatmap
    sns.heatmap(anterior_matrix, annot=True, cmap="Reds", fmt="d", cbar_kws={'label': 'Number of DILs'}, ax=axs[0])
    axs[0].set_title('Anterior Slice')
    axs[0].set_xticklabels(['Left', 'Right'])
    axs[0].set_yticklabels(['Base', 'Mid', 'Apex'])

    # Posterior heatmap
    sns.heatmap(posterior_matrix, annot=True, cmap="Blues", fmt="d", cbar_kws={'label': 'Number of DILs'}, ax=axs[1])
    axs[1].set_title('Posterior Slice')
    axs[1].set_xticklabels(['Left', 'Right'])
    axs[1].set_yticklabels(['Base', 'Mid', 'Apex'])

    plt.tight_layout()

    # Save the plot
    svg_dose_fig_name = general_plot_name_string + '.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    plt.close(fig)






















def production_plot_sum_to_one_tissue_class_binom_regression_matplotlib(pydicom_item,
                                                                                 patientUID,
                                                                                 bx_ref,
                                                                                 all_ref_key,
                                                                                 structs_referenced_dict,
                                                                                 default_exterior_tissue,
                                                                                 patient_sp_output_figures_dir_dict,
                                                                                 general_plot_name_string):


    def stacked_area_plot_with_confidence_intervals(patientUID,
                                                    bx_struct_roi,
                                                    df, 
                                                    stacking_order):
        """
        Create a stacked area plot for binomial estimator values with confidence intervals,
        stacking the areas to sum to 1 at each Z (Bx frame) point. Confidence intervals are 
        shown as black dotted lines, properly shifted to align with stacked lines.

        :param df: pandas DataFrame containing the data
        :param stacking_order: list of tissue class names, ordered by stacking hierarchy
        """
        plt.ioff()
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate a common x_range for plotting
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Initialize cumulative variables for stacking
        y_cumulative = np.zeros_like(x_range)

        # Set color palette for tissue classes
        colors = plt.cm.viridis(np.linspace(0, 1, len(stacking_order)))

        # Loop through the stacking order
        for i, tissue_class in enumerate(stacking_order):
            tissue_df = df[df['Tissue class'] == tissue_class]

            # Perform kernel regression for binomial estimator
            kr = KernelReg(endog=tissue_df['Binomial estimator'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            y_kr, _ = kr.fit(x_range)

            # Perform kernel regression for CI lower and upper bounds
            kr_lower = KernelReg(endog=tissue_df['CI lower vals'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            kr_upper = KernelReg(endog=tissue_df['CI upper vals'], exog=tissue_df['Z (Bx frame)'], var_type='c', bw=[1])
            ci_lower_kr, _ = kr_lower.fit(x_range)
            ci_upper_kr, _ = kr_upper.fit(x_range)

            # Stack the binomial estimator values (fill between previous and new values)
            ax.fill_between(x_range, y_cumulative, y_cumulative + y_kr, color=colors[i], alpha=0.7, label=tissue_class)

            # Plot the black dotted lines for confidence intervals, shifted by the cumulative values
            ax.plot(x_range, y_cumulative + ci_upper_kr, color='black', linestyle=':', linewidth=1)  # Upper confidence interval
            ax.plot(x_range, y_cumulative + ci_lower_kr, color='black', linestyle=':', linewidth=1)  # Lower confidence interval

            # Update cumulative binomial estimator for stacking
            y_cumulative += y_kr

        # Final plot adjustments
        ax.set_title(str(patientUID) + str(bx_struct_roi) + 'Stacked Binomial Estimator with Confidence Intervals by Tissue Class')
        ax.set_xlabel('Biopsy axial dimension (mm)')
        ax.set_ylabel('Binomial Estimator')
        ax.legend(loc='best', facecolor='white')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        return fig


    # plotting loop
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]

    tissue_heirarchy_list = misc_tools.tissue_heirarchy_list_tissue_names_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = True,
                                       default_exterior_tissue = default_exterior_tissue
                                       )

    multi_structure_mc_sum_to_one_pt_wise_results_dataframe = pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - sum-to-one mc results"]


    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
        bx_struct_roi = specific_bx_structure["ROI"]
        
        sp_structure_mc_sum_to_one_pt_wise_results_dataframe = multi_structure_mc_sum_to_one_pt_wise_results_dataframe[multi_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx index"] == specific_bx_structure_index]

        fig = stacked_area_plot_with_confidence_intervals(patientUID,
                                                    bx_struct_roi,
                                                    sp_structure_mc_sum_to_one_pt_wise_results_dataframe, 
                                                    tissue_heirarchy_list)

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

        fig.savefig(svg_dose_fig_file_path, format='svg')

        # clean up for memory
        plt.close(fig)






def production_plot_sum_to_one_tissue_class_nominal_plotly(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_ref,
                                                all_ref_key,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string
                                                ):

    def tissue_class_sum_to_one_nominal_plot(df, y_axis_order, patientID, bx_struct_roi):
        df = misc_tools.convert_categorical_columns(df, ['Tissue class', 'Nominal'], [str, int])

        # Generate a list of colors using viridis colormap in Matplotlib
        stacking_order = y_axis_order
        colors = plt.cm.viridis(np.linspace(0, 1, len(stacking_order)))  # Same method you used in Matplotlib

        # Convert the colors to a format Plotly understands (hex strings)
        hex_colors = ['#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]

        # Create a color mapping for tissue classes
        color_mapping = dict(zip(stacking_order, hex_colors))

        # Create the scatter plot and pass the custom color map
        fig = px.scatter(
            df, 
            x='Z (Bx frame)', 
            y='Tissue class', 
            size='Nominal',  # Size based on Nominal (0 or 1)
            size_max=10,  # Set size for the points that appear
            color='Tissue class',  # Use tissue class for color assignment
            color_discrete_map=color_mapping,  # Apply the custom color mapping
            title=f'Sum-to-one Nominal tissue class along biopsy major axis (Pt: {patientID}, Bx: {bx_struct_roi})'
        )

        # Customize point style
        fig.update_traces(
            marker=dict(
                symbol='x',  # Change to other shapes like 'diamond', 'square', etc.
                #line=dict(width=2, color='DarkSlateGrey'),  # Add border to points
                #size=12,  # Set a base size (adjustable)
                #opacity=1,  # Set point transparency
                #color = 'black'
            )
        )

        # Clear all existing legend entries
        fig.for_each_trace(lambda trace: trace.update(showlegend=False))

        # Add dummy scatter points for the legend with fixed size
        for tissue_class in stacking_order:
            fig.add_scatter(
                x=[None],  # Dummy invisible point
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color_mapping[tissue_class], symbol='x'),
                name=tissue_class,  # Ensure tissue class appears in legend
                showlegend=True
            )

        # Customize labels and make the plot flatter by tweaking y-axis category settings
        fig.update_layout(
            xaxis_title="Biopsy axial dimension (mm)",
            yaxis_title="Tissue class",
            yaxis={
                'categoryorder': 'array',  # Set custom order
                'categoryarray': y_axis_order,  # Use the provided order for categories
                'tickvals': y_axis_order,  # Ensure the ticks follow this order
                'tickmode': 'array',
                'ticktext': y_axis_order,
                'scaleanchor': "x",  # Lock the aspect ratio of x and y
                'dtick': 1,  # Control category spacing
            },
            height=400,  # Adjust the overall height of the plot to flatten it
            legend_title_text='Tissue class'  # Set legend title
        )

        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = True)


        return fig 


    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]


    # Define the specific order for the y-axis categories
    y_axis_order = ['DIL', 'Urethral', 'Rectal', 'Prostatic', 'Periprostatic']

    multi_structure_mc_sum_to_one_pt_wise_results_dataframe = pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - sum-to-one mc results"] 

    
    # Plotting loop
    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
        bx_struct_roi = specific_bx_structure["ROI"]
        
        mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = multi_structure_mc_sum_to_one_pt_wise_results_dataframe[multi_structure_mc_sum_to_one_pt_wise_results_dataframe["Bx index"] == specific_bx_structure_index]

        fig = tissue_class_sum_to_one_nominal_plot(mc_compiled_results_sum_to_one_for_fixed_bx_dataframe, y_axis_order, patientUID, bx_struct_roi)

        bx_sp_plot_name_string = f"{bx_struct_roi} - " + general_plot_name_string

        svg_dose_fig_name = bx_sp_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height/3) # added /3 here to make the y axis categories closer together, ie to make the plot flatter so that it can fit beneath the sum-to-one spatial regression plots.

        html_dose_fig_name = bx_sp_plot_name_string+'.html'
        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
        fig.write_html(html_dose_fig_file_path) 





def cohort_global_scores_boxplot(cohort_mc_sum_to_one_global_scores_dataframe,
                                 general_plot_name_string,
                                 cohort_output_figures_dir):

    df = cohort_mc_sum_to_one_global_scores_dataframe

    # Melt the DataFrame to bring mean, min, and max into a single column for easier plotting
    df_melted = pandas.melt(df, id_vars=['Tissue class'], value_vars=['Global Mean BE', 'Global Min BE', 'Global Max BE'], 
                        var_name='Statistic', value_name='Binomial Estimator')

    # Create a grouped boxplot using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))  # Capture the figure and axis objects
    sns.boxplot(x='Tissue class', y='Binomial Estimator', hue='Statistic', data=df_melted, palette="Set2", ax=ax)

    # Customize the plot for better aesthetics
    ax.set_title('Boxplots of Global Mean, Min, and Max (sum-to-one) Values by Tissue Class', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    svg_dose_fig_name = general_plot_name_string + '.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    plt.savefig(svg_dose_fig_file_path, format='svg')

    # Close the figure to release memory
    plt.close(fig)  # Ensure the figure is closed properly


def cohort_global_scores_boxplot_by_bx_type(cohort_mc_sum_to_one_global_scores_dataframe,
                                 general_plot_name_string,
                                 cohort_output_figures_dir):

    df = cohort_mc_sum_to_one_global_scores_dataframe

    # Melt the DataFrame to bring mean, min, and max into a single column for easier plotting
    df_melted = pandas.melt(df, id_vars=['Tissue class', 'Simulated type'], 
                            value_vars=['Global Mean BE', 'Global Min BE', 'Global Max BE'], 
                            var_name='Statistic', value_name='Binomial Estimator')

    # Create a grouped boxplot using seaborn with faceting by 'Simulated type'
    g = sns.catplot(x='Tissue class', y='Binomial Estimator', hue='Statistic', 
                    col='Simulated type', data=df_melted, kind='box', 
                    palette="Set2", height=6, aspect=1.5)

    # Set y-axis limits to be between 0 and 1
    g.set(ylim=(0, 1))

    # Add horizontal grid lines
    g.set_axis_labels("Tissue Class", "Binomial Estimator")
    for ax in g.axes.flat:
        ax.grid(True, which='both', axis='y')  # Add horizontal grid lines to each subplot

    # Customize the plot for better aesthetics
    g.set_titles("Simulated Type: {col_name}")
    g.fig.suptitle('Boxplots of Global Mean, Min, and Max (sum-to-one) Values by Tissue Class', y=1.02, fontsize=12)
    g.set_xticklabels(rotation=45)
    plt.tight_layout()

    # Save the figure
    svg_dose_fig_name = general_plot_name_string + '.svg'
    svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
    g.savefig(svg_dose_fig_file_path, format='svg')

    # Close the figure to release memory
    plt.close(g.fig)  # Ensure the figure is closed properly







def production_plot_cohort_sum_to_one_binom_est_ridge_plot_by_voxel(cohort_mc_sum_to_one_pt_wise_results_dataframe,
                                                     svg_image_width,
                                                     svg_image_height,
                                                     dpi,
                                                     ridge_line_tissue_class_general_plot_name_string,
                                                     cohort_output_figures_dir):

    plt.ioff()  # Turn off interactive plotting for batch figure generation

    df = copy.deepcopy(cohort_mc_sum_to_one_pt_wise_results_dataframe)
    df = misc_tools.convert_categorical_columns(df, ["Simulated type","Tissue class", "Binomial estimator", "Voxel index", "Voxel begin (Z)", "Voxel end (Z)"], [str, str, float, int, float, float])


    # Create a discrete colormap (similar to the desired color scheme)
    colors = ["green", "blue", "black"]
    cmap = LinearSegmentedColormap.from_list("GreenBlueBlack", colors, N=10)  # Discrete colors
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Set the theme for seaborn plots
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Main loop for plotting
    for bx_type, group_bx_type in df.groupby('Simulated type'):
        for tissue_class, group in group_bx_type.groupby('Tissue class'):
            
            unique_voxels = group['Voxel index'].unique()
            palette_black = {voxel: "black" for voxel in unique_voxels}  # Default color palette
            
            def annotate_and_color(x, color, label, **kwargs):
                specific_group = group[group['Voxel index'] == float(label)]
                if not specific_group.empty:
                    mean_prob = np.mean(x)
                    std_prob = np.std(x)

                    ax = plt.gca()

                    if np.std(x) < 1e-6 or len(np.unique(x)) <= 1:
                        # No variability, plot as a spike at 0
                        if np.all(x == 0):
                            ax.axvline(x=0, color="gray", linestyle='-', lw=2)
                            max_density_value = 0
                        else:
                            max_density_value = mean_prob
                    else:
                        try:
                            kde = gaussian_kde(x)
                            x_grid = np.linspace(0, 1, 1000)
                            y_density = kde(x_grid)
                            max_density_value = x_grid[np.argmax(y_density)]
                            
                            # Normalize density and plot
                            scaled_density = y_density / np.max(y_density)
                            density_color = cmap(norm(max_density_value))  # Color by max density
                            ax.fill_between(x_grid, scaled_density, alpha=0.5, color=density_color)

                        except np.linalg.LinAlgError:
                            max_density_value = "Error"

                    # Add vertical lines for mean, max density, and quantiles (Q05, Q25, Q75, Q95)
                    q05, q25, q75, q95 = np.percentile(x, [5, 25, 75, 95])  # Removed 50th quantile as it is the median
                    ax.axvline(x=max_density_value, color='magenta', linestyle='-', label='Max Density (Gy)', linewidth=2)
                    ax.axvline(x=mean_prob, color='orange', linestyle='-', label='Mean (Gy)', linewidth=2)
                    ax.axvline(x=np.median(x), color='red', linestyle='-', label='Median (Gy)', linewidth=2)  # Median only

                    # Plot quantiles with dashed lines (except 50th, handled as median)
                    for quantile_value in [q05, q25, q75, q95]:
                        ax.axvline(x=quantile_value, color='cyan', linestyle='--', linewidth=2)

                    # Annotate statistics
                    voxel_begin = specific_group['Voxel begin (Z)'].min()
                    voxel_end = specific_group['Voxel end (Z)'].max()

                    annotation_text = (f'Voxel segment (mm): ({voxel_begin:.1f}, {voxel_end:.1f})\n'
                                    f'Mean: {mean_prob:.2f}\nSD: {std_prob:.2f}\n'
                                    f'Max Density: {max_density_value:.2f}')
                    
                    # Move the annotation into the white space (outside the plot area)
                    ax.text(1.05, 0.6, annotation_text, horizontalalignment='left', 
                            verticalalignment='center', transform=ax.transAxes, 
                            color=color, fontsize=9)

            # Setup for each tissue class (FacetGrid)
            g = sns.FacetGrid(group, row="Voxel index", hue="Voxel index", aspect=8, height=1, palette=palette_black)
            g.map(annotate_and_color, "Binomial estimator")

            # Adjust layout to make space for color bar and legend on the left
            g.fig.subplots_adjust(left=0.2, right=0.65)  # Shift the plot slightly to the right

            # Move the colorbar closer to the plot, and rotate the label
            cbar_ax = g.fig.add_axes([0.08, 0.2, 0.03, 0.6])  # Color bar now closer to the plot
            colorbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
            colorbar.set_label("Max Density Value", rotation=90, labelpad=-60, ha='center')


            # Grid lines on the ridgeline plots
            for ax in g.axes.flat:
                ax.grid(True, which='both', axis='x', linestyle='-', color='gray', linewidth=0.5)
                ax.set_axisbelow(True)

            # Final adjustments
            g.set_titles("")
            g.set(yticks=[])
            g.despine(bottom=False, left=True)
            g.set_axis_labels("Binomial estimator distribution", "")  # Updated x-axis label

            # Position "Density" label between color bar and ridgeline plot
            g.fig.text(0.16, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)

            # Move the legend further left
            handles = [
                Line2D([0], [0], color='magenta', lw=2, label='Max Density'),
                Line2D([0], [0], color='orange', lw=2, label='Mean'),
                Line2D([0], [0], color='red', lw=2, label='Median'),
                Line2D([0], [0], color='cyan', lw=2, linestyle='--', label='Quantiles (05, 25, 75, 95)')
            ]
            legend = g.fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.4, 0.95), frameon=True)
            legend.get_frame().set_facecolor('white')  # Set the legend background to white

            # Title and figure size
            plt.suptitle(f'Tissue Class: {tissue_class}', fontsize=16, fontweight='bold', y=1.02)  # Adjusted title position

            # Calculate the figure size in inches for the desired dimensions in pixels
            figure_width_in = svg_image_width / dpi
            figure_height_in = svg_image_height / dpi
            g.fig.set_size_inches(figure_width_in, figure_height_in)

            # Save the figure
            svg_dose_fig_name = ridge_line_tissue_class_general_plot_name_string + f'_{bx_type}_{tissue_class}.svg'
            svg_dose_fig_file_path = cohort_output_figures_dir.joinpath(svg_dose_fig_name)
            g.fig.savefig(svg_dose_fig_file_path, format='svg', dpi=dpi, bbox_inches='tight')

            plt.close(g.fig)







def production_plot_cohort_sum_to_one_nominal_counts_voxel_vs_tissue_class_heatmap(dataframe,
                                       svg_image_width,
                                       svg_image_height,
                                       dpi,
                                       heatmap_plot_name_string,
                                       output_dir):
    
    df = copy.deepcopy(dataframe)
    df = misc_tools.convert_categorical_columns(df, ["Simulated type", "Tissue class", "Nominal", "Voxel index", "Voxel begin (Z)", "Voxel end (Z)"], [str, str, int, int, float, float])

    # Loop over each unique "Simulated type" to generate separate heatmaps
    for simulated_type, group_df in df.groupby('Simulated type'):
        
        # Group by 'Voxel index' and 'Tissue class' to get the sum of 'Nominal' counts for each pairing
        heatmap_data = group_df.groupby(['Voxel index', 'Tissue class'])['Nominal'].sum().unstack(fill_value=0)
        
        # Compute total counts per voxel across all tissue classes for row normalization
        voxel_totals = heatmap_data.sum(axis=1)
        
        # Normalize each box by the total count of its respective voxel row
        normalized_heatmap_data = heatmap_data.div(voxel_totals, axis=0).fillna(0)
        
        # Prepare annotations with both absolute and normalized counts
        annotations = np.empty_like(heatmap_data, dtype=object)
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                absolute = heatmap_data.iloc[i, j]
                normalized = normalized_heatmap_data.iloc[i, j]
                annotations[i, j] = f"{absolute}, {normalized:.2f}"  # Format as "absolute, normalized"
        
        # Retrieve minimum voxel begin and maximum voxel end values for each voxel index
        voxel_annotations = group_df.groupby('Voxel index').agg({'Voxel begin (Z)': 'min', 'Voxel end (Z)': 'max'})
        
        # Create the plot
        plt.ioff()  # Turn off interactive plotting for batch figure generation
        fig, ax = plt.subplots(figsize=(svg_image_width / dpi, svg_image_height / dpi), dpi=dpi)
        
        sns.heatmap(
            normalized_heatmap_data,
            annot=annotations,  # Display absolute and normalized counts
            fmt='',  # Allows custom annotation format
            cmap="YlGnBu",
            cbar_kws={'label': 'Normalized Counts (Proportion)'},
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            vmin=0,   # Set the minimum value for the color scale
            vmax=1    # Set the maximum value for the color scale
        )
        
        # Add text annotations for the total counts and voxel begin/end next to each voxel index
        for y, (voxel_index, total_count) in enumerate(voxel_totals.items()):
            voxel_begin = voxel_annotations.loc[voxel_index, 'Voxel begin (Z)']
            voxel_end = voxel_annotations.loc[voxel_index, 'Voxel end (Z)']
            ax.text(-0.5, y + 0.5, f"Total counts: {int(total_count)}\nVoxel segment (mm): ({voxel_begin:.1f}, {voxel_end:.1f})",
                    ha='right', va='center', color='black', fontsize=9)
        
        # Customize labels and title
        ax.set_xlabel("Tissue Class", fontsize=12)
        ax.set_ylabel("Voxel Index", fontsize=12)
        ax.set_title(f"Normalized Heatmap for '{simulated_type}' - Voxel Index vs Tissue Class vs Nominal counts", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save the plot
        output_filename = f"{heatmap_plot_name_string}_{simulated_type}.svg"
        output_path = output_dir.joinpath(output_filename)
        fig.savefig(output_path, format='svg', dpi=dpi, bbox_inches='tight')
        
        # Close the figure to free memory
        plt.close(fig)





def production_plot_cohort_sum_to_one_all_biopsy_voxels_binom_est_histogram_by_tissue_class(dataframe,
                                       svg_image_width,
                                       svg_image_height,
                                       dpi,
                                       histogram_plot_name_string,
                                       output_dir,
                                       bx_sample_pts_vol_element,
                                       bin_width=0.05,
                                       bandwidth=0.1):
    
    plt.ioff()  # Turn off interactive plotting for batch figure generation
    
    # Deep copy the dataframe to prevent modifications to the original data
    df = copy.deepcopy(dataframe)
    
    # Get the list of unique tissue classes
    tissue_classes = df['Tissue class'].unique()
    
    # Set up the figure and subplots for each tissue class
    fig, axes = plt.subplots(len(tissue_classes), 1, figsize=(svg_image_width / dpi, svg_image_height / dpi), dpi=dpi, sharex=True)
    
    # Increase padding between subplots
    fig.subplots_adjust(hspace=0.5)  # Adjust hspace to increase vertical padding
    
    # Create color mappings for vertical lines
    line_colors = {
        'mean': 'orange',
        'min': 'blue',
        'max': 'purple',
        'q05': 'cyan',
        'q25': 'green',
        'q75': 'green',
        'q95': 'cyan',
        'max density': 'magenta'
    }
    
    for ax, tissue_class in zip(axes, tissue_classes):
        tissue_data = df[df['Tissue class'] == tissue_class]['Binomial estimator'].dropna()
        
        count = len(tissue_data)
        ax.text(-0.2, 0.85, f'Num voxels: {count}', ha='left', va='top', transform=ax.transAxes, fontsize=10, color='black')
        ax.text(-0.2, 0.7, f'Kernel BW: {bandwidth}', ha='left', va='top', transform=ax.transAxes, fontsize=10, color='black')
        ax.text(-0.2, 0.55, f'Bin width: {bin_width}', ha='left', va='top', transform=ax.transAxes, fontsize=10, color='black')
        ax.text(-0.2, 0.4, f'Bx voxel volume (cmm): {bx_sample_pts_vol_element}', ha='left', va='top', transform=ax.transAxes, fontsize=10, color='black')


        bins = np.arange(0, 1.05, bin_width)  # Create bins from 0 to 1 with steps of 0.05

        # Plot normalized histogram with KDE
        sns.histplot(tissue_data, bins=bins, kde=False, color='skyblue', stat='density', ax=ax)

        # Calculate statistics
        mean_val = tissue_data.mean()
        min_val = tissue_data.min()
        max_val = tissue_data.max()
        quantiles = np.percentile(tissue_data, [5, 25, 75, 95])

        
        try:
            # KDE fit for the binomial estimator values with specified bandwidth
            kde = gaussian_kde(tissue_data, bw_method=bandwidth)
            x_grid = np.linspace(0, 1, 1000)
            y_density = kde(x_grid)
            # Normalize the KDE so the area under the curve equals 1
            y_density /= np.trapz(y_density, x_grid)  # Normalize over the x_grid range

            max_density_value = x_grid[np.argmax(y_density)]

            # Overlay KDE plot
            ax.plot(x_grid, y_density, color='black', linewidth=1.5, label='KDE')

        except np.linalg.LinAlgError as e:
            # If there's a LinAlgError, it likely means all values are identical
            print(f"Cohort sum-to-one histogram plot | Tissue class: {tissue_class} | LinAlgError: {e}")
            constant_value = tissue_data.iloc[0] if len(tissue_data) > 0 else 0
            ax.axvline(constant_value, color='black', linestyle='-', linewidth=1.5, label='All values are identical')
            max_density_value = constant_value  # Set max density to the constant value for further annotations

        except Exception as e:
            # Handle any other unexpected errors and print/log the error message
            print(f"Cohort sum-to-one histogram plot | Tissue class: {tissue_class} | An unexpected error occurred: {e}")
            # Set a fallback for max density value or other defaults
            constant_value = tissue_data.mean() if len(tissue_data) > 0 else 0
            ax.axvline(constant_value, color='red', linestyle='-', linewidth=1.5, label='Fallback line due to error')
            max_density_value = constant_value

        # Add vertical lines for mean, min, max, quantiles, and max density
        line_positions = {
            'Mean': mean_val,
            'Min': min_val,
            'Max': max_val,
            'Q05': quantiles[0],
            'Q25': quantiles[1],
            'Q75': quantiles[2],
            'Q95': quantiles[3],
            'Max Density': max_density_value
        }
        
                # Sort line_positions by the x-values (positions of the vertical lines)
        sorted_line_positions = sorted(line_positions.items(), key=lambda item: item[1])

        # Initialize tracking variables to handle overlapping labels
        last_x_val = None
        last_label_y = 1.02  # Initial y position for text labels
        stack_count = 0  # Track count of stacked labels
        offset_x = 0  # Horizontal offset for secondary stacks

        # Iterate over the sorted line positions to add vertical lines and labels
        for label, x_val in sorted_line_positions:
            color = line_colors.get(label.lower(), 'black')
            ax.axvline(x_val, color=color, linestyle='--' if 'Q' in label else '-', label=label)

            # Check for potential overlap and adjust y-position if needed
            if last_x_val is not None and abs(x_val - last_x_val) < 0.1:
                last_label_y += 0.1
                stack_count += 1
            else:
                # Reset position and stack count if no overlap
                last_label_y = 1.02
                stack_count = 0
                offset_x = 0

            # Shift label to the right if stack count exceeds 3
            if stack_count > 2:
                offset_x += 0.02  # Increment horizontal offset
                last_label_y = 1.02  # Reset y-position for the new stack
                stack_count = 0  # Reset stack count for the new column

            # Add text above the plot area with adjusted x and y positions
            ax.text(x_val + offset_x, last_label_y, f'{x_val:.2f}', color=color, ha='center', va='bottom',
                    fontsize=9, transform=ax.get_xaxis_transform())

            # Update last_x_val to current x_val
            last_x_val = x_val


        # Set x-axis limits to [0, 1] and enable grid lines
        ax.set_xlim(0, 1)
        ax.grid(True)
        ax.set_xticks(np.arange(0, 1.1, 0.1))  # Sets vertical grid lines every 0.1
        ax.set_xlabel('')

        # Add title and labels with adjusted title position
        ax.set_title(f'{tissue_class}', fontsize=12, y=1, x = -0.15, ha='left')
        ax.set_ylabel('Density')
        
    # X-axis label and figure title
    fig.text(0.5, 0.04, 'Binomial Estimator', ha='center', fontsize=12)
    fig.suptitle('Cohort - Normalized Binomial Estimator Distribution by Tissue Class For All Biopsy Voxels', fontsize=14)

    # Legend positioned outside the plot area with white background
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.95, 0.5), frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Save the figure
    output_path = output_dir.joinpath(f"{histogram_plot_name_string}.svg")
    fig.savefig(output_path, format='svg', dpi=dpi, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)






########## MR PLOTS
        
def production_plot_axial_mr_distribution_quantile_regression_by_patient_matplotlib(pydicom_item,
                                                                                 patientUID,
                                                                                 bx_ref,
                                                                                 mr_type_ref,
                                                                                 patient_sp_output_figures_dir_dict,
                                                                                 general_plot_name_string,
                                                                                 col_name_str_prefix,
                                                                                 output_dataframe_str):
    # plotting function
    def plot_quantile_regression_and_more_corrected(df, col_name_str, patientUID, bx_id):
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))

        # Generate a common x_range for plotting
        x_range = np.linspace(df['Z (Bx frame)'].min(), df['Z (Bx frame)'].max(), 500)

        # Placeholder dictionaries for regression results
        y_regressions = {}

        # Function to perform and plot kernel regression
        def perform_and_plot_kernel_regression(x, y, x_range, label, color):
            kr = KernelReg(endog=y, exog=x, var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            plt.plot(x_range, y_kr, label=label, color=color, linewidth=2)

        # Perform kernel regression for each quantile and store the y-values
        for quantile in [0.05, 0.25, 0.75, 0.95]:
            q_df = df.groupby('Z (Bx frame)')[col_name_str].quantile(quantile).reset_index()
            kr = KernelReg(endog=q_df[col_name_str], exog=q_df['Z (Bx frame)'], var_type='c', bw = [1])
            y_kr, _ = kr.fit(x_range)
            y_regressions[quantile] = y_kr

        # Filling the space between the quantile lines
        plt.fill_between(x_range, y_regressions[0.05], y_regressions[0.25], color='springgreen', alpha=1)
        plt.fill_between(x_range, y_regressions[0.25], y_regressions[0.75], color='dodgerblue', alpha=1)
        plt.fill_between(x_range, y_regressions[0.75], y_regressions[0.95], color='springgreen', alpha=1)
        
        # Additional plot enhancements
        # Plot line for MC trial num = 0
        # Kernel regression for MC trial num = 0 subset
        
        mc_trial_0 = df[df['MC trial num'] == 0]
        perform_and_plot_kernel_regression(mc_trial_0['Z (Bx frame)'], mc_trial_0[col_name_str], x_range, 'Nominal', 'red')
        

        # KDE and mean dose per Original pt index
        kde_max_doses = []
        mean_doses = []
        z_vals = []
        for z_val in df['Z (Bx frame)'].unique():
            pt_data = df[df['Z (Bx frame)'] == z_val]
            kde = gaussian_kde(pt_data[col_name_str])
            kde_doses = np.linspace(pt_data[col_name_str].min(), pt_data[col_name_str].max(), 500)
            max_density_dose = kde_doses[np.argmax(kde(kde_doses))]
            kde_max_doses.append(max_density_dose)
            mean_doses.append(pt_data[col_name_str].mean())
            z_vals.append(z_val)
        
        perform_and_plot_kernel_regression(z_vals, kde_max_doses, x_range, 'KDE Max Density', 'magenta')
        perform_and_plot_kernel_regression(z_vals, mean_doses, x_range, 'Mean', 'orange')

        num_mc_trials_plus_nom = df['MC trial num'].nunique()

        # Line plot for each trial
        for trial in range(1,num_mc_trials_plus_nom):
            df_sp_trial = df[df["MC trial num"] == trial].sort_values(by='Z (Bx frame)') # sorting is to make sure that the lines are drawn properly
            df_z_simple = df_sp_trial.drop_duplicates(subset=['Z (Bx frame)'], keep='first') # remove points that have the same z value so that the line plots look better
            plt.plot(df_z_simple['Z (Bx frame)'], df_z_simple[col_name_str], color='grey', alpha=0.1, linewidth=1, zorder = 0.9)  # 'linewidth' controls the thickness of the line, zorder puts these lines below the fill betweens!
        

        

        plt.title(f'Quantile Regression with Filled Areas Between Lines - {patientUID} - {bx_id}')
        plt.xlabel('Z (Bx frame)')
        plt.ylabel(col_name_str)
        plt.legend(['5th-25th Percentile', '25th-75th Percentile', '75th-95th Percentile', 'Nominal', 'Max density', 'Mean'], loc='best', facecolor = 'white')
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        return fig
    
    # plotting loop
    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]

    mr_type_subdict = pydicom_item[mr_type_ref]
    col_name_str = col_name_str_prefix + " " +str(mr_type_subdict["Units"])

    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
        bx_struct_roi = specific_bx_structure["ROI"]
        
        sp_bx_mr_distribution_all_trials_df = specific_bx_structure["Output data frames"][output_dataframe_str]

        fig = plot_quantile_regression_and_more_corrected(sp_bx_mr_distribution_all_trials_df, col_name_str, patientUID, bx_struct_roi)

        svg_dose_fig_name = bx_struct_roi + general_plot_name_string+'.svg'
        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)

        fig.savefig(svg_dose_fig_file_path, format='svg')

        # clean up for memory
        plt.close(fig)

















































#### HELPERS











def add_p_value_annotation(fig, 
                           array_columns, 
                           subplot=1,
                           show_p_val = True,
                           _format=dict(interline=0.07, text_height=1.07, color='black')
                           ):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare 
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        y_range[i] = [1.01+i*_format['interline'], 1.02+i*_format['interline']]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str =str(subplot)
        indices = [] #Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            #print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        #print((indices))
    else:
        subplot_str = ''

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        #print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        #print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        # Get the p-value
        pvalue = stats.ttest_ind(
            fig_dict['data'][data_pair[0]]['y'],
            fig_dict['data'][data_pair[1]]['y'],
            equal_var=False,
        )[1]
        if pvalue >= 0.05:
            symbol = 'ns'
        elif pvalue >= 0.01: 
            symbol = '*'
        elif pvalue >= 0.001:
            symbol = '**'
        else:
            symbol = '***'
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][0], 
            x1=column_pair[0], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Horizontal line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][1], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[1], y0=y_range[index][0], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'],size=14),
            x=(column_pair[0] + column_pair[1])/2,
            y=y_range[index][1]*_format['text_height'],
            showarrow=False,
            text=symbol,
            textangle=0,
            xref="x"+subplot_str,
            yref="y"+subplot_str+" domain"
        ))

        if show_p_val == True:
            fig.add_annotation(dict(font=dict(color=_format['color'],size=14),
            x=(column_pair[0] + column_pair[1])/2,
            y=y_range[index][1]*(_format['text_height']-0.02),
            showarrow=False,
            text='p = '+str(round(pvalue,5)),
            textangle=0,
            xref="x"+subplot_str,
            yref="y"+subplot_str+" domain"
        ))
    return fig



def add_p_value_annotation_v2(fig, array_columns, subplot=1, show_p_val=True, _format=dict(color='black')):
    '''
    Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: list
        list of which columns to compare 
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: int
        specifies the subplot number to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        y_range[i] = [1.01 + i * 0.07, 1.02 + i * 0.07]

    # Get indices if working with subplots
    if subplot:
        subplot_str = str(subplot) if subplot > 1 else ''
        indices = [i for i, data in enumerate(fig.data) if data['xaxis'] == 'x' + subplot_str]
    else:
        subplot_str = ''
        indices = list(range(len(fig.data)))

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        
        # Get the p-value
        y1_data = fig.data[data_pair[0]].y
        y2_data = fig.data[data_pair[1]].y
        pvalue = stats.ttest_ind(y1_data, y2_data, equal_var=False)[1]
        
        if pvalue >= 0.05:
            symbol = 'ns'
        elif pvalue >= 0.01:
            symbol = '*'
        elif pvalue >= 0.001:
            symbol = '**'
        else:
            symbol = '***'

        p_value_text = f'p = {pvalue:.1e} ({symbol})'
        x1, x2 = column_pair
        y0, y1 = y_range[index]

        # Estimate the width of the p-value text
        text_width = len(p_value_text) * 0.04  # Approximate width per character

        # Draw the bracket with dynamically calculated gap for p-value
        # Left vertical line
        fig.add_shape(type="line", xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=x1, y0=y0, x1=x1, y1=y1, line=dict(color=_format['color'], width=2))

        # Right vertical line
        fig.add_shape(type="line", xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=x2, y0=y0, x1=x2, y1=y1, line=dict(color=_format['color'], width=2))

        # Left horizontal line
        fig.add_shape(type="line", xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=x1, y0=y1, x1=(x1 + x2) / 2 - text_width / 2, y1=y1, line=dict(color=_format['color'], width=2))

        # Right horizontal line
        fig.add_shape(type="line", xref="x" + subplot_str, yref="y" + subplot_str + " domain",
                      x0=(x1 + x2) / 2 + text_width / 2, y0=y1, x1=x2, y1=y1, line=dict(color=_format['color'], width=2))

        # Add p-value text with significance symbol
        if show_p_val:
            fig.add_annotation(dict(font=dict(color=_format['color'], size=14),
                                    x=(x1 + x2) / 2,
                                    y=y1+0.04,
                                    showarrow=False,
                                    text=p_value_text,
                                    textangle=0,
                                    xref="x" + subplot_str,
                                    yref="y" + subplot_str + " domain",
                                    align="center",
                                    valign="middle"))
    return fig




def add_significance_annotation(fig, x=0.5, y=-0.1, _format=dict(color='black')):
    '''
    Adds an annotation to the figure explaining the significance symbols
    
    Parameters:
    ----------
    fig: figure
        plotly figure to which the annotation will be added
    x: float
        x position of the annotation in paper coordinates (default is 0.5, centered)
    y: float
        y position of the annotation in paper coordinates (default is -0.1, below the figure)
    _format: dict
        format characteristics for the text

    Returns:
    -------
    fig: figure
        figure with the added annotation
    '''
    
    # Significance explanation text
    significance_text = (
        "Significance symbols:\n"
        "ns: p ≥ 0.05\n"
        "*: 0.01 ≤ p < 0.05\n"
        "**: 0.001 ≤ p < 0.01\n"
        "***: p < 0.001"
    )
    
    # Add the annotation
    fig.add_annotation(
        text=significance_text,
        xref="paper", yref="paper",
        x=x, y=y,
        showarrow=False,
        font=dict(color=_format['color'], size=12),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )
    
    return fig
