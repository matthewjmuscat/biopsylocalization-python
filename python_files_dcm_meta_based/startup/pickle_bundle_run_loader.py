from dataclasses import dataclass
from pathlib import Path

from preprocessing.output_runtime_dirs import create_run_output_directories
from preprocessing.pickled_dataset_tools import load_pickle_bundle
from ui.tk_file_dialogs import askopenfilename_hidden_root


@dataclass(frozen=True)
class LoadedPickleBundleRun:
    master_structure_reference_dict: dict
    master_structure_info_dict: dict
    specific_output_dir: Path
    raw_mc_output_dir: Path
    reference_dict_path_str: str
    info_dict_path_str: str


def load_selected_pickle_bundle_run(reference_prompt,
                                    reference_title,
                                    info_prompt,
                                    info_title,
                                    initialdir,
                                    output_dir):
    print(reference_prompt)
    reference_dict_path_str = askopenfilename_hidden_root(
        title=reference_title,
        initialdir=initialdir,
    )

    print(info_prompt)
    reference_dict_path_parent = Path(reference_dict_path_str).parent
    info_dict_path_str = askopenfilename_hidden_root(
        title=info_title,
        initialdir=reference_dict_path_parent,
    )

    master_structure_reference_dict, master_structure_info_dict = load_pickle_bundle(
        reference_dict_path_str,
        info_dict_path_str,
    )
    specific_output_dir, raw_mc_output_dir = create_run_output_directories(
        master_structure_info_dict,
        output_dir,
    )

    return LoadedPickleBundleRun(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        specific_output_dir=specific_output_dir,
        raw_mc_output_dir=raw_mc_output_dir,
        reference_dict_path_str=reference_dict_path_str,
        info_dict_path_str=info_dict_path_str,
    )