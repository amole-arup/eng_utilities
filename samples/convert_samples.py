"""Generates list of files in the `samples` subdirectory, then
runs the parsing function (`run_all`), which generates the
`E2K_dict` dictionary containing all the model data,
and then runs the GWA export function on the `E2K_dict`.

The plan is to add this to the testing that should go on before accepting pull requests

TO DO
Some sort of accounting for converted quantities could be added.
"""

from sys import path as syspath
syspath.append(r'..\eng_utilities')

from os import listdir
from os.path import exists, isdir, isfile, join, basename, splitext
from unicodedata import east_asian_width
import pickle
from eng_utilities.E2K_parsing import run_all
from eng_utilities.GWA_utilities import write_GWA
from glob import glob
from fnmatch import fnmatch

# ========= U T I L I T I E S ===================

def file_list_maker2(item, patterns, recursive=False):
    """
    pattern is the pattern for the file
        e.g. r'*.[eE]2[kK]' to match model1.e2k, model2.E2K
             r'*_4.[eE]2[kK]' to match model_4.e2k, but not model4.E2K
    
    """

    file_list = []
    if isdir(item):
        print('Item is a directory')
        for pattern in patterns:
            file_list.extend([n for n in glob(item + '\\' + pattern, recursive=recursive)])
    elif isfile(item):
        print('Item is a file')
        for pattern in patterns:
            if fnmatch(item, pattern): # search(pattern, item):
                file_list.append(item)
    return file_list


def file_list_maker(files_or_dirs, patterns, recursive=False):

    file_list = []
    #log_dict = {}
    if isinstance(files_or_dirs, (list, tuple)):
        print('Collection of items')
        for item in files_or_dirs:
            file_list.extend(file_list_maker2(item, patterns, recursive=recursive))
                
    elif isinstance(files_or_dirs, str):
        print('Individual items')
        file_list.extend(file_list_maker2(files_or_dirs, patterns, recursive=recursive))
    else:
        pass
    
    return file_list


def run_list(e2k_list):
    
    log_dict = {}
    
    for eek in e2k_list[:]:  #  
        file_root, file_ext = splitext(eek)
        file_base = basename(file_root)
        file_dict = {}

        if exists(eek):
            print('\n--------------------------------')
            print(f'File found - {eek}')
            print('--------------------------------')
            file_dict = {'E2K': True}

            get_pickle = False

            try:
                print(f'\n...Parsing E2K - {eek} ')
                E2K_dict = run_all(eek, get_pickle=get_pickle, debug=False)
                print(f'\n...Completed parsing - {eek} ')
                if exists(file_root + '.pkl'):
                    file_dict['PKL'] = True

                if exists(file_root + '_2.pkl'):
                    file_dict['PKL2'] = True

                GWA_file = file_root + '.gwa'
                print(f'\n...writing to GWA - {GWA_file}')
                write_GWA(E2K_dict, GWA_file)
                print(('\n...Completed writing' if exists(GWA_file) else '\n...Failed to write') + f' to GWA - {GWA_file}')
                if exists(GWA_file):
                    file_dict['GWA'] = True

                file_dict['Complete'] = True
                 
            except:
                print(f'...!!! Processing failed - {eek} !!!')
                file_dict['Complete'] = False

        else:
            print('\n--------------------------------')
            print(f'\nFile NOT found - {eek}')
            print('\n--------------------------------')

        log_dict[file_base] = file_dict
    
    return log_dict


def tabular_print(data_dict, row_key, col_keys):
    divider = '===='
    res = ''.join([f'{k:6s}' for k in col_keys])
    headers = f'{row_key:40s}' + res
    print(headers)
    res = ''.join([f'{divider:6s}' for k in col_keys])
    headers = f'{divider:40s}' + res
    print(headers)

    for file, d in data_dict.items():
        res = ''.join([f'{str(d.get(k,False)):6s}' for k in col_keys])
        #file_len = len(file)
        file_len = sum(2 if east_asian_width(char) in "WF" else 1 for char in file)
        if file_len > 40:
            print(file)
            print(f'{"":40s}' + res)
        else:
            L = 40 - file_len + len(file)
            print(f'{file:{L}s}' + res)


def test_summary(data_dict, col_keys):
    return all(all([d.get(k,False) for k in col_keys]) for d in data_dict.values())
    


def test_samples(dir_path, print_summary=False, fake_run=False):
    """"""
    
    log_dict = {}
    # Set to True to test function
    # Set to False to run conversion checks on real files
    #fake_run = True

    if fake_run:
        if print_summary:
            print('\n** Fake data in use **')
        log_dict = {
            'Model 1': {'E2K': True, 'GWA': True, 'PKL': True, 'PKL2': True, 'Complete': True}, 
            'Model 3B': {'E2K': True, 'GWA': True, 'PKL': True, 'PKL2': True, 'Complete': False}, 
            'Model with an extremely long name that would mess up the formatting': {'E2K': True, 'GWA': True, 'PKL': True, 'PKL2': True, 'Complete': True}, 
            'Test model 04': {'E2K': True, 'GWA': False, 'PKL': True, 'PKL2': True, 'Complete': True}, 
            'Name with chinese characters 中文': {'E2K': True, 'GWA': False, 'PKL': True, 'PKL2': False, 'Complete': False},
            'Very long 中文 model name with chinese characters': {'E2K': True, 'GWA': False, 'PKL': True, 'PKL2': False, 'Complete': False},
            }
        e2k_list = []

    else:
        # Generation of list of model files in the specified directory
        e2k_list = file_list_maker(dir_path, (r'*.[eE]2[kK]', r'*.$[eE][tT]'), recursive=True)
        
        if print_summary:
            print(f'\nRunning analyses from \n\t{dir_path}')
        
        directory_listing = listdir(dir_path)
        e2k_list = [join(dir_path, fl) for fl in directory_listing if (fl.casefold().endswith('e2k') or fl.casefold().endswith('$et'))]
        
        if print_summary:
            print('____________________________\n')
            print('List length', len(e2k_list))
            [print(e2k) for e2k in e2k_list] # Read in the E2K text file

        # Run all analyses in the list
        log_dict = run_list(e2k_list)

    summary = test_summary(log_dict, ('E2K', 'GWA', 'PKL', 'PKL2', 'Complete'))

    if print_summary:
        print('\n=================================')    
        #print('log_dict: ')
        #print(log_dict)
        #print('=================================\n')    

        # Print results
        print('Overall Result is:')
        print('  ', summary, '\n')

        tabular_print(log_dict, 'File', ('E2K', 'GWA', 'PKL', 'PKL2', 'Complete'))

    return summary if not fake_run else False

# ========================================================


def main():
    # Specification of directory that contains model files
    dir_path = r'.\samples\StructuralModels'  # folders

    fake_run = True
    test_samples(dir_path, True, fake_run)


if __name__ == '__main__':
    main()