import pathlib
from datetime import datetime


DEFAULT_RAW_MC_OUTPUT_FOLDER_NAME = "Raw MC output"


def create_run_output_directories(master_structure_info_dict,
                                  output_dir,
                                  raw_mc_output_folder_name=DEFAULT_RAW_MC_OUTPUT_FOLDER_NAME):
    global_info = master_structure_info_dict.setdefault("Global", {})

    date_time_now = datetime.now()
    date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
    specific_output_dir_name = "MC_sim_out-" + date_time_now_file_name_format
    specific_output_dir = pathlib.Path(output_dir).joinpath(specific_output_dir_name)
    specific_output_dir.mkdir(parents=False, exist_ok=True)

    raw_mc_output_dir = specific_output_dir.joinpath(raw_mc_output_folder_name)
    raw_mc_output_dir.mkdir(parents=True, exist_ok=True)

    global_info["Specific output dir"] = specific_output_dir
    global_info["Raw MC output dir"] = raw_mc_output_dir

    return specific_output_dir, raw_mc_output_dir