"""Generates list of files in the `samples` subdirectory, then
runs the parsing function (`run_all`), which generates the
`E2K_dict` dictionary containing all the model data,
and then runs the GWA export function on the `E2K_dict`.
"""

from sys import path as syspath
syspath.append(r'..\eng_utilities')

from os import listdir
from os.path import exists, isfile, join, basename, splitext
import pickle
from eng_utilities.E2K_parsing import run_all
from eng_utilities.GWA_utilities import write_GWA

directory_listing = listdir(r'.\samples')
# join(f'..\samples', fl)
# print(directory_listing)
e2k_list = [join(f'.\samples', fl) for fl in directory_listing if (fl.casefold().endswith('e2k') or fl.casefold().endswith('$et'))]
[print(e2k) for e2k in e2k_list] # Read in the E2K text file

for eek in e2k_list[:]:
    if exists(eek):
        print('\n--------------------------------')
        print(f'File found - {eek}')
        print('--------------------------------')
    else:
        print(f'\nFile NOT found - {eek}')
        
    E2K_dict = run_all(eek, get_pickle=True)
    
    GWA_file = eek.replace(".e2k",".gwa").replace(".$et",".gwa").replace(".E2K",".gwa").replace(".$ET",".gwa")
    write_GWA(E2K_dict, GWA_file)
    

    pkl2_file = eek.replace(".e2k","_2.pkl").replace(".$et","_2.pkl").replace(".E2K","_2.pkl").replace(".$ET","_2.pkl")
    print(f'\n=== Use these lines to import the model data: ===')
    print(f'import pickle')
    print(f'E2K_dict = pickle.load(open({pkl2_file}, "rb")')   
