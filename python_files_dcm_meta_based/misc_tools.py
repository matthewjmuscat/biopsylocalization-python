import ques_funcs
import sys

def checkdirs(live_display, important_info, *paths):
    created_a_dir = False
    for path in paths:
        if path.exists():
            important_info.add_text_line(str(path)+ " already exists.", live_display)
        else:
            path.mkdir(parents=True, exist_ok=True)
            important_info.add_text_line("Path "+ str(path)+ " created.", live_display)
            created_a_dir = True
    if created_a_dir == True:
        live_display.stop()
        print('Directories have been created, please ensure the input folder is non-empty, then continue.')
        continue_programme = ques_funcs.ask_ok('> Continue?' )
        if continue_programme == False:
            sys.exit('> Programme exited.')
        else:
            live_display.start()


def find_closest_z_slice(threeD_data_zslice_list,z_val):
    # used to find the closest zslice of points to a given z value within the ThreeDdata structure 
    # which is a list of numpy arrays where each
    # element of the list is a constant zslice
    closest_z_slice_index = min(range(len(threeD_data_zslice_list)), key=lambda i: abs(threeD_data_zslice_list[i][0,2]-z_val))
    return closest_z_slice_index


