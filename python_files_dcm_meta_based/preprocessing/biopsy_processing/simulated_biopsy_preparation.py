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
		"Preparation complete": False,
	}


def _get_simulated_biopsy_preparation_dict(specific_structure):
	if specific_structure.get("Simulated biopsy preparation dict") is None:
		specific_structure["Simulated biopsy preparation dict"] = _create_default_simulated_biopsy_preparation_dict()

	return specific_structure["Simulated biopsy preparation dict"]


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
		dil_centroids_by_ref = {}
		if dil_ref in pydicom_item:
			for specific_dil_structure in pydicom_item[dil_ref]:
				dil_refnum = specific_dil_structure["Ref #"]
				dil_centroid = np.array(specific_dil_structure["Structure global centroid"]).reshape(3)
				dil_centroids_by_ref[dil_refnum] = dil_centroid

		for specific_structure in pydicom_item[bx_ref]:
			contour_length_mm = specific_structure.get("Reconstructed biopsy cylinder length (from contour data)")
			centroid_line_length_mm = specific_structure.get("Centroid line vec length (bx needle base to bx needle tip)")

			if specific_structure["Simulated bool"] == True:
				if centroid_line_length_mm is not None:
					_set_length_information(specific_structure,
											centroid_line_length_mm,
											None,
											contour_length_mm=contour_length_mm,
											centroid_line_length_mm=centroid_line_length_mm)
					simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
					simulated_biopsy_preparation_dict["Length determined"] = False
					simulated_biopsy_preparation_dict["Length source"] = None
					simulated_biopsy_preparation_dict["Nominal length mm"] = None
				continue

			if contour_length_mm is None:
				continue

			_set_length_information(specific_structure,
									contour_length_mm,
									"real contour reconstruction",
									contour_length_mm=contour_length_mm,
									centroid_line_length_mm=centroid_line_length_mm)

			real_biopsy_lengths_list.append(float(contour_length_mm))

			if dil_centroids_by_ref:
				bx_centroid = np.array(specific_structure["Structure global centroid"]).reshape(3)
				best_refnum = _find_nearest_dil_refnum(bx_centroid,
													  dil_centroids_by_ref)
				if best_refnum is not None:
					real_bx_lengths_by_dil[patientUID][best_refnum].append(float(contour_length_mm))

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
				length_mm = float(mean_of_real_biopsy_lengths)
				length_source = 'match real'

				relative_structure_type = specific_structure["Relative structure type"]
				relative_structure_refnum = specific_structure["Relative structure ref #"]
				if relative_structure_type == dil_ref:
					lengths_for_this_dil = real_bx_lengths_by_dil.get(patientUID, {}).get(relative_structure_refnum, [])
					if lengths_for_this_dil:
						length_mm = float(np.mean(lengths_for_this_dil))
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
										 live_display
										 ):
	for _, pydicom_item in master_structure_reference_dict.items():
		for specific_structure in pydicom_item[bx_ref]:
			simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)

			if specific_structure["Simulated bool"] == True:
				simulated_biopsy_preparation_dict["Multiplicity"] = 1
				simulated_biopsy_preparation_dict["Preparation complete"] = bool(
					simulated_biopsy_preparation_dict["Length determined"]
					and simulated_biopsy_preparation_dict["Target determined"]
				)
			else:
				simulated_biopsy_preparation_dict["Multiplicity"] = None
				simulated_biopsy_preparation_dict["Preparation complete"] = False

	return live_display


def simulated_biopsy_preparation_dataframe_builder(master_structure_reference_dict,
												   bx_ref,
												   all_ref_key,
												   live_display
												   ):
	cohort_simulated_biopsy_preparation_dataframe = pandas.DataFrame()

	for patientUID, pydicom_item in master_structure_reference_dict.items():
		patient_rows = []

		for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
			simulated_biopsy_preparation_dict = _get_simulated_biopsy_preparation_dict(specific_structure)
			patient_rows.append({
				"Patient ID": patientUID,
				"Bx index": specific_structure_index,
				"Bx ROI": specific_structure["ROI"],
				"Bx ref #": specific_structure["Ref #"],
				"Simulated bool": specific_structure["Simulated bool"],
				"Simulated type": specific_structure["Simulated type"],
				"Relative structure type": specific_structure.get("Relative structure type"),
				"Relative structure ref #": specific_structure.get("Relative structure ref #"),
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
				"Preparation complete": simulated_biopsy_preparation_dict["Preparation complete"],
			})

		sp_patient_simulated_biopsy_preparation_dataframe = pandas.DataFrame(patient_rows)
		pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Simulated biopsy preparation dataframe"] = sp_patient_simulated_biopsy_preparation_dataframe
		cohort_simulated_biopsy_preparation_dataframe = pandas.concat([
			cohort_simulated_biopsy_preparation_dataframe,
			sp_patient_simulated_biopsy_preparation_dataframe,
		], ignore_index=True)

	return cohort_simulated_biopsy_preparation_dataframe, live_display


def simulated_biopsy_preparer(master_structure_reference_dict,
							  bx_ref,
							  dil_ref,
							  all_ref_key,
							  simulated_biopsy_length_method,
							  biopsy_needle_compartment_length,
							  live_display
							  ):
	length_results_dict, live_display = determine_simulated_biopsy_lengths(master_structure_reference_dict,
																			bx_ref,
																			dil_ref,
																			simulated_biopsy_length_method,
																			biopsy_needle_compartment_length,
																			live_display)

	live_display = assign_simulated_biopsy_targets(master_structure_reference_dict,
												   bx_ref,
												   dil_ref,
												   live_display)

	live_display = expand_simulated_biopsy_multiplicity(master_structure_reference_dict,
														bx_ref,
														live_display)

	cohort_simulated_biopsy_preparation_dataframe, live_display = simulated_biopsy_preparation_dataframe_builder(master_structure_reference_dict,
																												   bx_ref,
																												   all_ref_key,
																												   live_display)

	length_results_dict["cohort_simulated_biopsy_preparation_dataframe"] = cohort_simulated_biopsy_preparation_dataframe

	return length_results_dict, live_display
