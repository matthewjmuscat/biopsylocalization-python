import copy
from collections import defaultdict

import numpy as np
import pandas


def _create_default_simulated_biopsy_preparation_dict():
	return {
		"Length determined": False,
		"Length source": None,
		"Contour length mm": None,
		"Centroid line length mm": None,
		"Nominal length mm": None,
		"Target determined": False,
		"Target source": None,
		"Target structure type": None,
		"Target structure ref #": None,
		"Target structure index": None,
		"Target structure ID": None,
		"Multiplicity": None,
		"Multiplicity index": None,
		"Real matched biopsy count": None,
		"Matched real biopsy ROI": None,
		"Matched real biopsy ref #": None,
		"Matched real biopsy index": None,
		"Extra biopsy bool": None,
		"Family source": None,
		"Multiplicity base ROI": None,
		"Multiplicity base ref #": None,
		"Preparation complete": False,
	}


def _get_simulated_biopsy_preparation_dict(specific_structure):
	if specific_structure.get("Simulated biopsy preparation dict") is None:
		specific_structure["Simulated biopsy preparation dict"] = _create_default_simulated_biopsy_preparation_dict()

	return specific_structure["Simulated biopsy preparation dict"]


def _update_preparation_complete(specific_structure):
	simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)

	if specific_structure["Simulated bool"] == True:
		simulated_biopsy_preparation_dict["Preparation complete"] = bool(
			simulated_biopsy_preparation_dict["Length determined"]
			and simulated_biopsy_preparation_dict["Target determined"]
			and simulated_biopsy_preparation_dict["Multiplicity"] is not None
			and simulated_biopsy_preparation_dict["Multiplicity index"] is not None
		)
	else:
		simulated_biopsy_preparation_dict["Preparation complete"] = False


def _find_nearest_dil_refnum(bx_centroid,
							 dil_centroids_by_ref
							 ):
	best_refnum = None
	best_dist2 = None
	for dil_refnum, dil_centroid in dil_centroids_by_ref.items():
		dist2 = np.sum((bx_centroid - dil_centroid) ** 2)
		if best_dist2 is None or dist2 < best_dist2:
			best_dist2 = dist2
			best_refnum = dil_refnum

	return best_refnum


def _find_structure_info_from_refnum(pydicom_item,
							 structure_type,
							 structure_refnum
							 ):
	if structure_type not in pydicom_item:
		return None, None

	for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
		if specific_structure["Ref #"] == structure_refnum:
			return specific_structure_index, specific_structure

	return None, None


def _find_real_biopsy_from_refnum(pydicom_item,
							  bx_ref,
							  real_biopsy_refnum
							  ):
	for specific_structure in pydicom_item[bx_ref]:
		if specific_structure["Simulated bool"] == False and specific_structure["Ref #"] == real_biopsy_refnum:
			return specific_structure

	return None


def _set_length_information(specific_structure,
							length_mm,
							length_source,
							contour_length_mm=None,
							centroid_line_length_mm=None
							):
	simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
	simulated_biopsy_preparation_dict["Length determined"] = True
	simulated_biopsy_preparation_dict["Length source"] = length_source
	simulated_biopsy_preparation_dict["Nominal length mm"] = float(length_mm)

	if contour_length_mm is not None:
		simulated_biopsy_preparation_dict["Contour length mm"] = float(contour_length_mm)

	if centroid_line_length_mm is not None:
		simulated_biopsy_preparation_dict["Centroid line length mm"] = float(centroid_line_length_mm)

	_update_preparation_complete(specific_structure)


def _set_target_information(specific_structure,
							target_determined,
							target_source,
							target_structure_type,
							target_structure_refnum,
							target_structure_index,
							target_structure_id
							):
	simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
	simulated_biopsy_preparation_dict["Target determined"] = target_determined
	simulated_biopsy_preparation_dict["Target source"] = target_source
	simulated_biopsy_preparation_dict["Target structure type"] = target_structure_type
	simulated_biopsy_preparation_dict["Target structure ref #"] = target_structure_refnum
	simulated_biopsy_preparation_dict["Target structure index"] = target_structure_index
	simulated_biopsy_preparation_dict["Target structure ID"] = target_structure_id

	_update_preparation_complete(specific_structure)


def _set_multiplicity_information(specific_structure,
							  multiplicity,
							  multiplicity_index,
							  real_matched_biopsy_count,
							  matched_real_biopsy_roi,
							  matched_real_biopsy_refnum,
							  matched_real_biopsy_index,
							  extra_biopsy_bool,
							  family_source,
							  multiplicity_base_roi,
							  multiplicity_base_refnum
							  ):
	simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
	simulated_biopsy_preparation_dict["Multiplicity"] = multiplicity
	simulated_biopsy_preparation_dict["Multiplicity index"] = multiplicity_index
	simulated_biopsy_preparation_dict["Real matched biopsy count"] = real_matched_biopsy_count
	simulated_biopsy_preparation_dict["Matched real biopsy ROI"] = matched_real_biopsy_roi
	simulated_biopsy_preparation_dict["Matched real biopsy ref #"] = matched_real_biopsy_refnum
	simulated_biopsy_preparation_dict["Matched real biopsy index"] = matched_real_biopsy_index
	simulated_biopsy_preparation_dict["Extra biopsy bool"] = extra_biopsy_bool
	simulated_biopsy_preparation_dict["Family source"] = family_source
	simulated_biopsy_preparation_dict["Multiplicity base ROI"] = multiplicity_base_roi
	simulated_biopsy_preparation_dict["Multiplicity base ref #"] = multiplicity_base_refnum

	_update_preparation_complete(specific_structure)


def _build_real_biopsy_matches_by_dil(pydicom_item,
							  bx_ref,
							  dil_ref
							  ):
	real_biopsy_matches_by_dil = defaultdict(list)

	for specific_structure in pydicom_item[bx_ref]:
		if specific_structure["Simulated bool"] == True:
			continue

		simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
		target_structure_type = simulated_biopsy_preparation_dict.get("Target structure type")
		target_structure_refnum = simulated_biopsy_preparation_dict.get("Target structure ref #")

		if target_structure_type == dil_ref and target_structure_refnum is not None:
			real_biopsy_matches_by_dil[target_structure_refnum].append({
				"ROI": specific_structure["ROI"],
				"Ref #": specific_structure["Ref #"],
				"Index number": specific_structure["Index number"],
			})

	for target_structure_refnum in real_biopsy_matches_by_dil.keys():
		real_biopsy_matches_by_dil[target_structure_refnum] = sorted(real_biopsy_matches_by_dil[target_structure_refnum],
													 key=lambda x: x["Index number"])

	return dict(real_biopsy_matches_by_dil)


def _get_base_identifier_value(specific_structure,
							   key_name,
							   base_key_name
							   ):
	simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
	base_value = simulated_biopsy_preparation_dict.get(base_key_name)
	if base_value is None:
		base_value = specific_structure[key_name]
		simulated_biopsy_preparation_dict[base_key_name] = base_value

	return base_value


def _format_simulated_identifier(base_value,
							 multiplicity_index
							 ):
	if multiplicity_index == 1:
		return base_value

	return "{} [{}]".format(base_value, multiplicity_index)


def _renumber_biopsy_indices(bx_structures):
	for specific_structure_index, specific_structure in enumerate(bx_structures):
		specific_structure["Index number"] = specific_structure_index


def _refresh_biopsy_info_dict(master_structure_info_dict,
							  patientUID,
							  bx_ref,
							  bx_structures
							  ):
	if master_structure_info_dict is None:
		return

	if master_structure_info_dict.get("By patient") is None:
		return

	if patientUID not in master_structure_info_dict["By patient"]:
		return

	if bx_ref not in master_structure_info_dict["By patient"][patientUID]:
		return

	num_sim_structs = sum(1 for specific_structure in bx_structures if specific_structure["Simulated bool"] == True)
	num_real_structs = len(bx_structures) - num_sim_structs
	bpsy_type_counts_dict = {}
	for specific_structure in bx_structures:
		simulated_type = specific_structure["Simulated type"]
		bpsy_type_counts_dict[simulated_type] = bpsy_type_counts_dict.get(simulated_type, 0) + 1

	master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"] = len(bx_structures)
	master_structure_info_dict["By patient"][patientUID][bx_ref]["Num sim structs"] = num_sim_structs
	master_structure_info_dict["By patient"][patientUID][bx_ref]["Num real structs"] = num_real_structs
	master_structure_info_dict["By patient"][patientUID][bx_ref]["Biopsy type counts"] = bpsy_type_counts_dict


def _refresh_global_biopsy_info_dict(master_structure_info_dict,
								 bx_ref
								 ):
	if master_structure_info_dict is None:
		return

	if master_structure_info_dict.get("Global") is None:
		return

	if master_structure_info_dict.get("By patient") is None:
		return

	global_num_biopsies = 0
	global_num_biopsies_by_type = {}

	for patientUID, patient_info_dict in master_structure_info_dict["By patient"].items():
		if bx_ref not in patient_info_dict:
			continue

		sp_patient_biopsy_info_dict = patient_info_dict[bx_ref]
		global_num_biopsies = global_num_biopsies + sp_patient_biopsy_info_dict.get("Num structs", 0)

		for biopsy_type, count in sp_patient_biopsy_info_dict.get("Biopsy type counts", {}).items():
			global_num_biopsies_by_type[biopsy_type] = global_num_biopsies_by_type.get(biopsy_type, 0) + count

	master_structure_info_dict["Global"]["Num biopsies"] = global_num_biopsies
	master_structure_info_dict["Global"]["Num biopsies by bx type dict"] = global_num_biopsies_by_type


def assign_simulated_biopsy_targets(master_structure_reference_dict,
									bx_ref,
									dil_ref,
									live_display
									):
	for patientUID, pydicom_item in master_structure_reference_dict.items():
		dil_centroids_by_ref = {}
		if dil_ref in pydicom_item:
			for specific_dil_structure in pydicom_item[dil_ref]:
				dil_refnum = specific_dil_structure["Ref #"]
				dil_centroid = np.array(specific_dil_structure["Structure global centroid"]).reshape(3)
				dil_centroids_by_ref[dil_refnum] = dil_centroid

		for specific_structure in pydicom_item[bx_ref]:
			if specific_structure["Simulated bool"] == False:
				target_determined = False
				target_source = None
				target_structure_type = None
				target_structure_refnum = None
				target_structure_index = None
				target_structure_id = None

				if dil_centroids_by_ref and specific_structure.get("Structure global centroid") is not None:
					bx_centroid = np.array(specific_structure["Structure global centroid"]).reshape(3)
					target_structure_refnum = _find_nearest_dil_refnum(bx_centroid,
											   dil_centroids_by_ref)
					if target_structure_refnum is not None:
						target_structure_type = dil_ref
						target_structure_index, target_structure = _find_structure_info_from_refnum(pydicom_item,
																	 dil_ref,
																	 target_structure_refnum)
						if target_structure is not None:
							target_determined = True
							target_source = "Nearest DIL by centroid"
							target_structure_id = target_structure["ROI"]

				_set_target_information(specific_structure,
									target_determined,
									target_source,
									target_structure_type,
									target_structure_refnum,
									target_structure_index,
									target_structure_id)
				continue

			relative_structure_type = specific_structure.get("Relative structure type")
			relative_structure_refnum = specific_structure.get("Relative structure ref #")
			target_determined = False
			target_source = None
			target_structure_index = None
			target_structure_id = None

			if relative_structure_type == dil_ref and relative_structure_refnum is not None:
				target_structure_index, target_structure = _find_structure_info_from_refnum(pydicom_item,
																 relative_structure_type,
																 relative_structure_refnum)
				if target_structure is not None:
					target_determined = True
					target_source = "Relative structure"
					target_structure_id = target_structure["ROI"]

			_set_target_information(specific_structure,
									target_determined,
									target_source,
									relative_structure_type,
									relative_structure_refnum,
									target_structure_index,
									target_structure_id)

	return live_display


def expand_simulated_biopsy_multiplicity(master_structure_reference_dict,
									 bx_ref,
									 dil_ref,
									 live_display,
									 master_structure_info_dict=None
									 ):
	for patientUID, pydicom_item in master_structure_reference_dict.items():
		real_biopsy_matches_by_dil = _build_real_biopsy_matches_by_dil(pydicom_item,
														 bx_ref,
														 dil_ref)

		real_biopsy_structures = []
		simulated_biopsy_groups_dict = {}
		simulated_biopsy_group_order = []

		for specific_structure in pydicom_item[bx_ref]:
			if specific_structure["Simulated bool"] == False:
				real_biopsy_structures.append(specific_structure)
				_set_multiplicity_information(specific_structure,
									  None,
									  None,
									  None,
									  None,
									  None,
									  None,
									  None,
									  None,
									  None,
									  None)
				continue

			simulated_group_key = (
				specific_structure["Simulated type"],
				specific_structure.get("Relative structure type"),
				specific_structure.get("Relative structure ref #"),
			)
			if simulated_group_key not in simulated_biopsy_groups_dict:
				simulated_biopsy_groups_dict[simulated_group_key] = []
				simulated_biopsy_group_order.append(simulated_group_key)

			simulated_biopsy_groups_dict[simulated_group_key].append(specific_structure)

		expanded_simulated_biopsy_structures = []

		for simulated_group_key in simulated_biopsy_group_order:
			specific_structures = simulated_biopsy_groups_dict[simulated_group_key]
			template_structure = specific_structures[0]
			relative_structure_type = template_structure.get("Relative structure type")
			relative_structure_refnum = template_structure.get("Relative structure ref #")

			if relative_structure_type == dil_ref and relative_structure_refnum is not None:
				matched_real_biopsy_rows = real_biopsy_matches_by_dil.get(relative_structure_refnum, [])
				requested_multiplicity = max(len(matched_real_biopsy_rows), 1)
			else:
				matched_real_biopsy_rows = []
				requested_multiplicity = max(len(specific_structures), 1)

			selected_structures = list(specific_structures[:requested_multiplicity])
			while len(selected_structures) < requested_multiplicity:
				selected_structures.append(copy.deepcopy(template_structure))

			for multiplicity_index, specific_structure in enumerate(selected_structures, start=1):
				matched_real_biopsy_row = None
				if multiplicity_index <= len(matched_real_biopsy_rows):
					matched_real_biopsy_row = matched_real_biopsy_rows[multiplicity_index - 1]

				multiplicity_base_roi = _get_base_identifier_value(specific_structure,
													  "ROI",
													  "Multiplicity base ROI")
				multiplicity_base_refnum = _get_base_identifier_value(specific_structure,
														  "Ref #",
														  "Multiplicity base ref #")

				specific_structure["ROI"] = _format_simulated_identifier(multiplicity_base_roi,
														  multiplicity_index)
				specific_structure["Ref #"] = _format_simulated_identifier(multiplicity_base_refnum,
															 multiplicity_index)

				if matched_real_biopsy_row is not None:
					matched_real_biopsy_roi = matched_real_biopsy_row["ROI"]
					matched_real_biopsy_refnum = matched_real_biopsy_row["Ref #"]
					matched_real_biopsy_index = matched_real_biopsy_row["Index number"]
					extra_biopsy_bool = False
					family_source = "Matched real biopsy"
				elif relative_structure_type == dil_ref and relative_structure_refnum is not None:
					matched_real_biopsy_roi = None
					matched_real_biopsy_refnum = None
					matched_real_biopsy_index = None
					extra_biopsy_bool = True
					family_source = "Extra DIL without matched real biopsy"
				else:
					matched_real_biopsy_roi = None
					matched_real_biopsy_refnum = None
					matched_real_biopsy_index = None
					extra_biopsy_bool = None
					family_source = None

				_set_multiplicity_information(specific_structure,
									  requested_multiplicity,
									  multiplicity_index,
									  len(matched_real_biopsy_rows),
									  matched_real_biopsy_roi,
									  matched_real_biopsy_refnum,
									  matched_real_biopsy_index,
									  extra_biopsy_bool,
									  family_source,
									  multiplicity_base_roi,
									  multiplicity_base_refnum)

				expanded_simulated_biopsy_structures.append(specific_structure)

		pydicom_item[bx_ref] = real_biopsy_structures + expanded_simulated_biopsy_structures
		_renumber_biopsy_indices(pydicom_item[bx_ref])
		_refresh_biopsy_info_dict(master_structure_info_dict,
									  patientUID,
									  bx_ref,
									  pydicom_item[bx_ref])

	_refresh_global_biopsy_info_dict(master_structure_info_dict,
								 bx_ref)

	return live_display


def determine_simulated_biopsy_lengths(master_structure_reference_dict,
									   bx_ref,
									   dil_ref,
									   simulated_biopsy_length_method,
									   biopsy_needle_compartment_length,
									   live_display
									   ):
	real_biopsy_lengths_list = []
	real_bx_lengths_by_dil = defaultdict(lambda: defaultdict(list))

	for patientUID, pydicom_item in master_structure_reference_dict.items():
		for specific_structure in pydicom_item[bx_ref]:
			contour_length_mm = specific_structure.get("Reconstructed biopsy cylinder length (from contour data)")
			centroid_line_length_mm = specific_structure.get("Centroid line vec length (bx needle base to bx needle tip)")

			if specific_structure["Simulated bool"] == True:
				simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
				if contour_length_mm is not None:
					simulated_biopsy_preparation_dict["Contour length mm"] = float(contour_length_mm)
				if centroid_line_length_mm is not None:
					simulated_biopsy_preparation_dict["Centroid line length mm"] = float(centroid_line_length_mm)
				simulated_biopsy_preparation_dict["Length determined"] = False
				simulated_biopsy_preparation_dict["Length source"] = None
				simulated_biopsy_preparation_dict["Nominal length mm"] = None
				_update_preparation_complete(specific_structure)
				continue

			if contour_length_mm is None:
				continue

			_set_length_information(specific_structure,
									contour_length_mm,
									"real contour reconstruction",
									contour_length_mm=contour_length_mm,
									centroid_line_length_mm=centroid_line_length_mm)

			real_biopsy_lengths_list.append(float(contour_length_mm))

			simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
			target_structure_type = simulated_biopsy_preparation_dict.get("Target structure type")
			target_structure_refnum = simulated_biopsy_preparation_dict.get("Target structure ref #")
			if target_structure_type == dil_ref and target_structure_refnum is not None:
				real_bx_lengths_by_dil[patientUID][target_structure_refnum].append(float(contour_length_mm))

	if len(real_biopsy_lengths_list) >= 1:
		real_biopsy_lengths_arr = np.array(real_biopsy_lengths_list, dtype=float)
		mean_of_real_biopsy_lengths = float(np.mean(real_biopsy_lengths_arr))
		std_of_real_biopsy_lengths = float(np.std(real_biopsy_lengths_arr))
	else:
		mean_of_real_biopsy_lengths = float(biopsy_needle_compartment_length)
		std_of_real_biopsy_lengths = 0.0

	for patientUID, pydicom_item in master_structure_reference_dict.items():
		for specific_structure in pydicom_item[bx_ref]:
			if specific_structure["Simulated bool"] == False:
				continue

			if simulated_biopsy_length_method == 'full':
				length_mm = float(biopsy_needle_compartment_length)
				length_source = 'full'
			elif simulated_biopsy_length_method == 'real normal':
				within_bounds = False
				while within_bounds == False:
					length_mm = np.random.normal(loc=mean_of_real_biopsy_lengths,
										 scale=std_of_real_biopsy_lengths)
					if std_of_real_biopsy_lengths == 0:
						within_bounds = True
					elif (length_mm >= mean_of_real_biopsy_lengths - 2 * std_of_real_biopsy_lengths) and (length_mm <= mean_of_real_biopsy_lengths + 2 * std_of_real_biopsy_lengths):
						within_bounds = True
				length_mm = float(length_mm)
				length_source = 'real normal'
			elif simulated_biopsy_length_method == 'real mean':
				length_mm = float(mean_of_real_biopsy_lengths)
				length_source = 'real mean'
			elif simulated_biopsy_length_method == 'match real':
				simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
				matched_real_biopsy_refnum = simulated_biopsy_preparation_dict.get("Matched real biopsy ref #")
				matched_real_biopsy = None
				if matched_real_biopsy_refnum is not None:
					matched_real_biopsy = _find_real_biopsy_from_refnum(pydicom_item,
														   bx_ref,
														   matched_real_biopsy_refnum)

				if matched_real_biopsy is not None and matched_real_biopsy.get("Reconstructed biopsy cylinder length (from contour data)") is not None:
					length_mm = float(matched_real_biopsy["Reconstructed biopsy cylinder length (from contour data)"])
					length_source = 'match real - matched biopsy'
				else:
					length_mm = float(mean_of_real_biopsy_lengths)
					relative_structure_type = specific_structure.get("Relative structure type")
					relative_structure_refnum = specific_structure.get("Relative structure ref #")
					if relative_structure_type == dil_ref:
						lengths_for_this_dil = real_bx_lengths_by_dil.get(patientUID, {}).get(relative_structure_refnum, [])
						if lengths_for_this_dil:
							length_mm = float(np.mean(lengths_for_this_dil))
							length_source = 'match real - DIL mean'
						else:
							length_source = 'match real - global mean'
					else:
						length_source = 'match real - global mean'
			else:
				length_mm = float(biopsy_needle_compartment_length)
				length_source = 'full'

			contour_length_mm = specific_structure.get("Reconstructed biopsy cylinder length (from contour data)")
			centroid_line_length_mm = specific_structure.get("Centroid line vec length (bx needle base to bx needle tip)")
			_set_length_information(specific_structure,
									length_mm,
									length_source,
									contour_length_mm=contour_length_mm,
									centroid_line_length_mm=centroid_line_length_mm)

	real_bx_lengths_by_dil_standard_dict = {}
	for patientUID, patient_lengths_dict in real_bx_lengths_by_dil.items():
		real_bx_lengths_by_dil_standard_dict[patientUID] = dict(patient_lengths_dict)

	length_results_dict = {
		"real_biopsy_lengths_list": real_biopsy_lengths_list,
		"real_bx_lengths_by_dil": real_bx_lengths_by_dil_standard_dict,
		"mean_of_real_biopsy_lengths": mean_of_real_biopsy_lengths,
		"std_of_real_biopsy_lengths": std_of_real_biopsy_lengths,
	}

	return length_results_dict, live_display


def simulated_biopsy_preparation_dataframe_builder(master_structure_reference_dict,
										   bx_ref,
										   all_ref_key,
										   live_display
										   ):
	for patientUID, pydicom_item in master_structure_reference_dict.items():
		patient_rows = []

		for specific_structure in pydicom_item[bx_ref]:
			simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
			patient_rows.append({
				"Patient ID": patientUID,
				"Bx index": specific_structure["Index number"],
				"Bx ROI": specific_structure["ROI"],
				"Bx ref #": specific_structure["Ref #"],
				"Simulated bool": specific_structure["Simulated bool"],
				"Simulated type": specific_structure["Simulated type"],
				"Relative structure type": specific_structure.get("Relative structure type"),
				"Relative structure ref #": specific_structure.get("Relative structure ref #"),
				"Length determined": simulated_biopsy_preparation_dict["Length determined"],
				"Contour length mm": simulated_biopsy_preparation_dict["Contour length mm"],
				"Centroid line length mm": simulated_biopsy_preparation_dict["Centroid line length mm"],
				"Nominal length mm": simulated_biopsy_preparation_dict["Nominal length mm"],
				"Length source": simulated_biopsy_preparation_dict["Length source"],
				"Target determined": simulated_biopsy_preparation_dict["Target determined"],
				"Target source": simulated_biopsy_preparation_dict["Target source"],
				"Target structure type": simulated_biopsy_preparation_dict["Target structure type"],
				"Target structure ref #": simulated_biopsy_preparation_dict["Target structure ref #"],
				"Target structure index": simulated_biopsy_preparation_dict["Target structure index"],
				"Target structure ID": simulated_biopsy_preparation_dict["Target structure ID"],
				"Multiplicity": simulated_biopsy_preparation_dict["Multiplicity"],
				"Multiplicity index": simulated_biopsy_preparation_dict["Multiplicity index"],
				"Real matched biopsy count": simulated_biopsy_preparation_dict["Real matched biopsy count"],
				"Matched real biopsy ROI": simulated_biopsy_preparation_dict["Matched real biopsy ROI"],
				"Matched real biopsy ref #": simulated_biopsy_preparation_dict["Matched real biopsy ref #"],
				"Matched real biopsy index": simulated_biopsy_preparation_dict["Matched real biopsy index"],
				"Extra biopsy bool": simulated_biopsy_preparation_dict["Extra biopsy bool"],
				"Family source": simulated_biopsy_preparation_dict["Family source"],
				"Multiplicity base ROI": simulated_biopsy_preparation_dict["Multiplicity base ROI"],
				"Multiplicity base ref #": simulated_biopsy_preparation_dict["Multiplicity base ref #"],
				"Preparation complete": simulated_biopsy_preparation_dict["Preparation complete"],
			})

		sp_patient_simulated_biopsy_preparation_dataframe = pandas.DataFrame(patient_rows)
		pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Simulated biopsy preparation dataframe"] = sp_patient_simulated_biopsy_preparation_dataframe

	return live_display
def simulated_biopsy_preparer(master_structure_reference_dict,
							  bx_ref,
							  dil_ref,
							  all_ref_key,
							  simulated_biopsy_length_method,
							  biopsy_needle_compartment_length,
							  live_display,
							  master_structure_info_dict=None
							  ):
	live_display = assign_simulated_biopsy_targets(master_structure_reference_dict,
											   bx_ref,
											   dil_ref,
											   live_display)

	live_display = expand_simulated_biopsy_multiplicity(master_structure_reference_dict,
												bx_ref,
												dil_ref,
												live_display,
												master_structure_info_dict=master_structure_info_dict)

	_, live_display = determine_simulated_biopsy_lengths(master_structure_reference_dict,
															bx_ref,
															dil_ref,
															simulated_biopsy_length_method,
															biopsy_needle_compartment_length,
															live_display)

	live_display = simulated_biopsy_preparation_dataframe_builder(master_structure_reference_dict,
																   bx_ref,
																   all_ref_key,
																   live_display)

	return live_display
