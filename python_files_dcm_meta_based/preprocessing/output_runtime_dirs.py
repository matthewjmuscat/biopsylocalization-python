import pathlib
import json
import os
from datetime import datetime


DEFAULT_RAW_MC_OUTPUT_FOLDER_NAME = "Raw MC output"
RUN_COMPLETE_MANIFEST_FILENAME = "RUN_COMPLETE.json"


def _sanitize_output_dir_label(run_label):
    if run_label is None:
        return None

    label = str(run_label).strip()
    if len(label) == 0:
        return None

    allowed_chars = []
    for character in label:
        if character.isalnum() or character in {" ", "-", "_", ","}:
            allowed_chars.append(character)
        else:
            allowed_chars.append("-")
    sanitized_label = "".join(allowed_chars)
    while "  " in sanitized_label:
        sanitized_label = sanitized_label.replace("  ", " ")
    while "--" in sanitized_label:
        sanitized_label = sanitized_label.replace("--", "-")
    return sanitized_label.strip(" -_")


def create_run_output_directories(master_structure_info_dict,
                                  output_dir,
                                  raw_mc_output_folder_name=DEFAULT_RAW_MC_OUTPUT_FOLDER_NAME,
                                  run_label=None,
                                  run_metadata=None):
    global_info = master_structure_info_dict.setdefault("Global", {})

    date_time_now = datetime.now()
    date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
    specific_output_dir_name = "MC_sim_out-" + date_time_now_file_name_format
    sanitized_run_label = _sanitize_output_dir_label(run_label)
    if sanitized_run_label is not None:
        specific_output_dir_name = specific_output_dir_name + " - " + sanitized_run_label
    specific_output_dir = pathlib.Path(output_dir).joinpath(specific_output_dir_name)
    specific_output_dir.mkdir(parents=False, exist_ok=True)

    raw_mc_output_dir = specific_output_dir.joinpath(raw_mc_output_folder_name)
    raw_mc_output_dir.mkdir(parents=True, exist_ok=True)

    global_info["Specific output dir"] = specific_output_dir
    global_info["Raw MC output dir"] = raw_mc_output_dir
    global_info["Run output folder label"] = sanitized_run_label
    global_info["Run output metadata"] = {} if run_metadata is None else dict(run_metadata)

    return specific_output_dir, raw_mc_output_dir


def write_run_completion_manifest(
        *,
        output_dir,
        master_structure_info_dict,
        status="completed",
        message="Programme complete."):
    output_dir = pathlib.Path(output_dir)
    manifest_path = output_dir.joinpath(RUN_COMPLETE_MANIFEST_FILENAME)
    temporary_manifest_path = output_dir.joinpath(RUN_COMPLETE_MANIFEST_FILENAME + ".tmp")
    global_info = master_structure_info_dict.get("Global", {})
    payload = {
        "status": status,
        "message": message,
        "completed_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "specific_output_dir": str(global_info.get("Specific output dir", output_dir)),
        "raw_mc_output_dir": str(global_info.get("Raw MC output dir", "")),
        "run_output_folder_label": global_info.get("Run output folder label"),
        "run_output_metadata": global_info.get("Run output metadata", {}),
        "num_cases": global_info.get("Num cases"),
        "num_structures": global_info.get("Num structures"),
    }
    with temporary_manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    os.replace(temporary_manifest_path, manifest_path)
    return manifest_path