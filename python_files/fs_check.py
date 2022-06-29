# to check the file system to ensure that the data is in the correct structure and type

import pathlib
from termcolor import colored
import textwrap


def fs_checker(Data_name,detailed_output):
    """
    The main file system checker function, checks the data directory
    to make sure the data folder is properly structured, checks for existence, emptiness.
    Note that the messages are accumulated throughout the programme, numbered, labelled and then
    returned at the end to be printed by the calling programme.
    """

    fsc_failed = False # initially set failed variable to false
    messages = [] # create an empty list of messages
    # access the path two levels up, this is where the data folder should be located relative to the main programme
    try:
        Path_two_levels_up = pathlib.Path(__file__).parents[2]
    except:
        print("Error: perhaps you do not have permission to two levels up?")
    else:
        messages.append([len(messages),'info','Accessing structure two levels up'])

    # Check to ensure a folder specified by the variable Data_name exists in the correct location
    dir_two_levels_up = [x for x in Path_two_levels_up.iterdir() if x.is_dir() == True]
    for p in dir_two_levels_up:
        if p.name == Data_name:
            Data_exists = True
            break
        else:
            Data_exists = False
            pass
    
    # Handle the output of the data folder check
    if Data_exists == True:
        messages.append([len(messages),'success','The Data folder seems to exist in the correct place.'])
    else:
        fail_message = 'The Data folder may be misnamed or not exist in the correct location.'
        messages.append([len(messages),'fail',fail_message])
        fsc_failed = True
        return fsc_failed, messages

    # Try joining the Data folder, I feel like this may throw an error if permissions are insufficient, but this try catch may be useless...
    try:
        data_folder = Path_two_levels_up.joinpath(Data_name)
    except:
        print("Error: perhaps you do not have permission to access the data folder?")
    else:
        messages.append([len(messages),'info',"Accessing data folder"])

    # check the data folder for its files and directories, the directories should be a list of patients, any loose files are ignored.
    num_data_directories, num_data_files, data_empty_message = test_empty(data_folder)
    
    # handle the output of the Data directory check, and determine the messages to pass on
    if int(num_data_directories) == 0:
        fsc_failed = True
        fail_message = 'Your ..\\Data folder contains no patient folders'
        messages.append([len(messages),'fail', fail_message])
        return fsc_failed, messages
    elif int(num_data_directories) != 0 and int(num_data_files) != 0:
        data_info_message = data_empty_message
        data_warning_message = 'There are '+str(num_data_files)+' loose files in the ..\\Data folder, these will be ignored!'
        messages.append([len(messages), 'warning', data_warning_message]) 
        messages.append([len(messages), 'info', data_info_message])
    elif int(num_data_directories) != 0 and int(num_data_files) == 0:
        data_info_message = data_empty_message
        messages.append([len(messages),'info',data_info_message])

    """
    # check each patient folder under Data, for any patient folders that are empty, stop the programme and tell the user
    patients_total_treatments = 0
    patients_total_loose_files = 0
    empty_patient = []
    for patient in data_folder.iterdir():
        num_ptnt_directories, num_ptnt_files, ptnt_empty_message = test_empty(patient)
        if int(num_ptnt_directories) == 0:
            empty_patient.append(patient) 
        patients_total_loose_files = patients_total_loose_files + int(num_ptnt_files)
        patients_total_treatments = patients_total_treatments + int(num_ptnt_directories)

    # handle the output of the patient folders check
    if len(empty_patient) != 0:
        fsc_failed = True
        fail_message = 'The following directories contain no treatment sub-folders:\n'+'\n'.join([str(x) for x in empty_patient])
        return fsc_failed, fail_message
    elif len(empty_patient) == 0:
        print('Found '+str(patients_total_treatments)+' treatment subfolders')
        if patients_total_loose_files != 0:
            print(colored('Warning:','yellow'), str(patients_total_loose_files)+ ' loose files found in treatment subfolders, these will be ignored!')
    """
    
    # check each patient folder for emptiness by passing the data folder to 
    # the function directory_checker, since this function checks two levels down 
    bool_a_data_folder_empty, messages = directory_checker(data_folder,detailed_output,messages)
    
    # handle the output of the patient folder check and determine if the overall
    # file system check has passed or failed
    if bool_a_data_folder_empty == True:
        fsc_failed = bool_a_data_folder_empty
        return fsc_failed, messages
    
    # pass all patient folders as directorys to the directory_checker function to determine
    # if all treatment subfolders are empty or not, they should each contain folders of 
    passed_message = [len(messages),'success','FSC passed!']
    messages.append(passed_message)
    return fsc_failed, messages


def test_empty(directory):
    """
    A function to test whether or not a directory is empty, and return 
    whether that directory contains files and/or folders, and how many 
    of each. This programme assumes that the directory to be searched exists.
    """
    directories = [x for x in directory.iterdir() if x.is_dir()]
    files = [x for x in directory.iterdir() if x.is_file()]
    num_directories = str(len(directories))
    num_files = str(len(files))

   
    empty_message = 'The ..\\'+ directory.name+' directory contains '+num_directories+' directories and '+num_files+' files.'

    return num_directories, num_files, empty_message


def directory_checker(directory,detailed_output,messages_list = []):
    """
    A function that checks all subdirectories in the passed directory, for folders and files 
    and outputs messages accordingly. The number of files and folders that each subdirectory 
    contains is only output as a message if the detailed_output boolean is True. Note the 
    function also accepts a messages list as an argument, but has a default value of null
    if no message list is passed to it.
    """
    total_num_folders = 0
    total_num_files = 0
    empty_folders = []
    subdirectories_list = [x for x in directory.iterdir() if x.is_dir()]
    for folder in subdirectories_list:
        local_num_folders, local_num_files, test_empty_message = test_empty(folder)
        if int(local_num_folders) == 0:
            empty_folders.append(folder) 
        total_num_files = total_num_files + int(local_num_files)
        total_num_folders = total_num_folders + int(local_num_folders)
        if detailed_output == True:
            messages_list.append([len(messages_list),'info',test_empty_message])

    # handle the output of the patient folders check, 
    # fail if some directories under the passed directory contain no subdirectories
    # pass if they do.
    if len(empty_folders) != 0:
        bool_empty = True
        message = [len(messages_list),'fail','The following directories contain no sub-directories:\n'+'\n'.join([textwrap.indent(str(x),'+   ') for x in empty_folders])]
    elif len(empty_folders) == 0:
        bool_empty = False
        message = [len(messages_list),'success','Found '+str(total_num_folders)+' sub-directories beneath (..\\'+subdirectories_list[0].name+ ' <---> ..\\'+ subdirectories_list[-1].name+'), all of them are non-empty.']
        
    messages_list.append(message)
    
    # include a warning message if there are loose files
    if total_num_files != 0:
        warningmsg = [len(messages_list),'warning',str(total_num_files)+ ' loose files found immediately beneath '+str(len(subdirectories_list))+' subdirectories (..\\'+subdirectories_list[0].name+ ' <---> ..\\'+ subdirectories_list[-1].name+') these will be ignored!']
        messages_list.append(warningmsg)
    return bool_empty, messages_list


def main():
    print('This is the file system check programme, it is not meant to be run directly.')

if __name__ == '__main__': 
    main()