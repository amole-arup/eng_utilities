""""""

from os import listdir
from os.path import exists, isfile, join, basename, splitext
import xmltodict
import pickle
import sys
import os

from eng_utilities.general_utilities import try_numeric, units_conv_dict, units_lookup_dict 
from eng_utilities.geometry_utilities import *

#sys.path.append(f'{os.path.dirname(__file__)}')  # /. ??


prop_dims_dict = {
    'B': 1, 'BF': 1, 'TF': 1, 'D': 1, 'OD': 1, 'ID': 1,
    'TDES': 1, 'HT': 1, 'X': 1, 'Y': 1, 'KDES': 1,
    'A': 2, 'AREACORE': 2, 'AS2': 2, 'AS3': 2,
    'I22': 4, 'I33': 4, 'J': 4, 
    'Z22': 3, 'Z33': 3, 'S33POS': 3, 'S33NEG': 3, 'S22POS': 3, 'S22NEG': 3, 
    'R33': 1,  'R22': 1, 'DIS': 1,
}


def convert_prop_units(shape_dict, model_length_units):
    """Converts section properties into model units
    
    Note that this will not carry out conversions 
    if there is no 'UNITS' defined in the shape, or
    if no conversion is defined, e.g. for forces into length(!)
    
    Args:
        shape_dict (dict): should contain at least area `A` and 
            length units `UNITS`, e.g. {'A':2.1, 'UNITS':'m'}
        model_length_units (str): the length units in the model
    """
    #print('model_length_units (convert_prop_units): \n ', model_length_units)
    #print('shape_dict (convert_prop_units): \n ', shape_dict)
    
    if isinstance(shape_dict, dict):
        # Get the units from the shape, default to 'm'
        prop_units = shape_dict.get('UNITS')
        if prop_units:
            prop_conv = units_conv_dict.get((prop_units, model_length_units), None)
            # Carry out conversion if the property name is in the 
            #  property dictionary - prop_dims_dict
            if prop_units.casefold() == model_length_units.casefold():
                return {k:try_numeric(v) \
                    for k, v in shape_dict.items() if prop_dims_dict.get(k)}
            elif prop_conv:
                return {k:(try_numeric(v) * prop_conv**prop_dims_dict[k]) \
                    for k, v in shape_dict.items() if prop_dims_dict.get(k)}
            else:
                print(f'call to units_conv_dict failed with: ({prop_units}, {model_length_units})')
                print('units_conv_dict: ', units_conv_dict)


## ==========================
## === Section Properties ===
## ==========================

def section_file_parser(root_dir, file_list = [], flatten=False, flat=False):
    """
    
    root_dir = r'C:\Program Files\Computers and Structures\ETABS 17'
    dir1 = 'Property Libraries Old'
    dir2 = 'Property Libraries'
    
    """
    if len(file_list) == 0:
        directory_listing = listdir(root_dir)
        xml_list = [fl for fl in directory_listing if fl.endswith('xml')]
    else:
        xml_list = file_list
    
    sec_d4 = dict()
    for j, xml_file in enumerate(xml_list):
        #xml_file = 'AISC14.xml'
        #print(f'{j}: {xml_file}')
        
        filename = basename(xml_file).replace('.xml','') if flatten else xml_file.replace('.xml','')
        with open(join(root_dir, xml_file), 'rb') as f:
            data = xmltodict.parse(f)
        
        if not data.get('PROPERTY_FILE'):
            continue
        
        units = data['PROPERTY_FILE']['CONTROL']['LENGTH_UNITS']
        # fix the capitalisation
        units = units_lookup_dict.get(units.casefold())
        exclude = ['AUTO_SELECT_LIST', 'GENERAL']
        table_keys = list(data['PROPERTY_FILE'].keys())
        if j<2: print(f'table_keys: {table_keys}')
        i = table_keys.index('CONTROL')
        #print('index: ', i)
        sections = [t for t in table_keys[i+1:] if not (t in exclude)]
        if j<2: print(f'sections: {sections}')
        sec_gen = ((k3, v3) for k3, v3 in data['PROPERTY_FILE'].items() if k3 in sections)

        sec_d2 = dict()
        for k2, v2 in sec_gen:
            #print(f'key k2: {k2}, v2 type: {type(v2)}')
            #sec_d1 = dict()
            if isinstance(v2, dict):
                v2 = [v2] # Convert single cases to list of one case
            for sec in v2: # a list
                sec_d0 = dict()
                sec_d0['SECTION_TYPE'] = k2
                for k,v in sec.items():
                    sec_d0[k] = try_numeric(v)
                sec_d0['UNITS'] = units
                sec_d2[sec_d0['LABEL']] = sec_d0
                if sec_d0.get('EDI_STD'):
                    sec_d2[sec_d0['EDI_STD']] = sec_d0
            #sec_d2[k2] = sec_d1

        #filename
        if flat: # creates a flat dictionary - needs testing
            sec_d4.update({(filename, k):v for k, v in sec_d2.items() 
                if ((v.get('DESIGNATION') != 'RB') and (v.get('DESIGNATION')) )})
        else:
            sec_d3 = {'FILENAME': filename, 
                    'CONTROL': data['PROPERTY_FILE']['CONTROL'],
                    'SECTIONS': sec_d2}
            sec_d4[filename] = sec_d3
    return sec_d4


def build_section_dict(ETABS_root_dir=None, flat = False, reimport = False, debug=False):
    """Generates a section dictionary from the ETABS section database
    unless a pickled dictionary already exists (can be overriden). 

    The section dictionary created will be stored as a pickle file for future reference

    Args:
        root_directory: this is the root directory for the ETABS installation. If this 
            is not specified, then it will look in the standard locations:
            `C:\Program Files\Computers and Structures\ETABS nn`
        flat: this will make a flat directory (not fully tested)
        reimport: this will force the generation of a new dictionary
            from the ETABS section files, even if a pickle file exists.

    """
    # This checks for a pickled section file and returns it if available
    picklePath = join(os.path.dirname(__file__), 'section_dict.pkl')
    if debug:
        print(picklePath)
        print('\nSection File ' + ('exists:' if exists(picklePath) else 'NOT found:') + \
            f' {picklePath}\n')
    if exists(picklePath) and not reimport: # Switch reimport to True to re-import
        return pickle.load(open(picklePath, 'rb'))
    
    if ETABS_root_dir is not None:
        # if the specified directory does not exist, then set path to `None`
        if not exists(ETABS_root_dir):
            ETABS_root_dir = None
    
    if os.name != 'nt':
        print('OS is not Windows and no section file found (section_dict.pkl)')
        return {}
    elif ETABS_root_dir is None:
        # Directory containing section data has not been provided
        CSi_dir = r'C:\Program Files\Computers and Structures'
        ETABS_dirs = [folder for folder in listdir(CSi_dir) if folder.startswith('ETABS')]
        
        if len(ETABS_dirs) == 0:
            return {}
        for ETABS_dir in ETABS_dirs:
            if exists(join(CSi_dir, ETABS_dirs[-1],'Property Libraries')):
                ETABS_root_dir = join(CSi_dir, ETABS_dir)
        if ETABS_root_dir is None:
            return {}
    
    dir1 = 'Property Libraries Old'
    dir2 = 'Property Libraries'
    file_list1 = listdir(join(ETABS_root_dir, dir1))
    xml_list1 = [fl for fl in file_list1 if fl.endswith('xml')]
    file_list2 = listdir(join(ETABS_root_dir, dir2))
    xml_list2 = [fl for fl in file_list2 if fl.endswith('xml')]
    xml_list = [join(dir1, file) for file in xml_list1] + \
            [join(dir2, file) for file in xml_list2]
    # Process section data
    section_def_dict = section_file_parser(ETABS_root_dir, xml_list, flatten=True, flat=flat)
    # Do pickle dump for future access
    pickle.dump(section_def_dict, open(picklePath, 'wb'))
        
    return section_def_dict


## ==========================
## === Section  Utilities ===
## ==========================

def A_props(D, B, TF, TW, ETABS=True):
    """Properties of single angle"""
    A = B * D - (D-TF) * (B-TW)
    Cz = 0.5 * (D**2 * TW + TF**2 * (B - TW)) / A
    Cy = 0.5 * (B**2 * TF + TW**2 * (D - TF)) / A
    Iyy = (TW * D**3 + B * TF**3 - TW*TF**3) / 3 - A * Cz**2
    Izz = (TF * B**3 + D * TW**3 - TF*TW**3) / 3 - A * Cy**2
    Iyz = (TW**2 * D**2 + B**2 * TF**2 - TW**2*TF**2) / 4 - A * Cy * Cz
    theta = 0.5 * atan2(-2 * Iyz,(Iyy - Izz))
    #theta = 0.5 * atan2(-2 * Iyz,(Izz - Iyy))
    p = 0.5 * (Iyy+Izz)
    q = (0.25 * (Iyy-Izz)**2 + Iyz**2)**0.5
    I1, I2 = p - q, p + q
    if ETABS:
        return {'P': 2*(B+D), 'C3': Cy, 'C2': Cz, 
            'A': A, 'AS3': (B-TW)*TF, 'AS2': (D-TF)*TW, 
            'I33': Iyy, 'I22': Izz, 'I23': Iyz, 
            'Iuu': I1, 'Ivv': I2, 'theta_rad':theta}
    else:
        return {'P': 2*(B+D), 'Cy': Cy, 'Cz': Cz, 
            'A': A, 'Avy': (B-TW)*TF, 'Avz': (D-TF)*TW, 
            'Iyy': Iyy, 'Izz': Izz, 'Iyz': Iyz, 
            'Iuu': I1, 'Ivv': I2, 'theta_rad':theta}


def AA_props(D, B, TF, TW, DIS=0, ETABS=True):
    """Properties of double angle
    Note that B is the sum of two legs + the distance between 
    the angles (DIS)
    """
    B0 = 0.5 * (B - DIS)
    A = 2 * (B0 * D - (D-TF) * (B0 - TW))
    Cz = (D**2 * TW + TF**2 * (B0 - TW)) / A
    Cy = 0
    Iyy = 2 * ((TW * D**3 + B0 * TF**3 - TW*TF**3) / 3 - A * Cz**2)
    Izz = ((TF * B0**3 + D * TW**3 - TF*TW**3) / 3 - A * (0.5*DIS)**2)
    if ETABS:
        return {'P': 2*(B0+D), 'C3': Cy, 'C2': Cz, 
            'A': A, 'AS3': 2*(B0-TW)*TF, 'AS2': 2*(D-TF)*TW, 
            'I33': Iyy, 'I22': Izz}
    else:
        return {'P': 2*(B0+D), 'Cy': Cy, 'Cz': Cz, 
            'A': A, 'Avy': 2*(B0-TW)*TF, 'Avz': 2*(D-TF)*TW, 
            'Iyy': Iyy, 'Izz': Izz}


def T_props(D, B, TF, TW, ETABS=True):
    """Properties of a Tee
    """
    # Cz dimensions initially measured from top,
    # but corrected in dictionary
    A = (B - TW) * TF + D * TW
    Cz = 0.5 * ((B - TW) * TF**2 + D**2 * TW) / A
    Cy = 0
    Iyy =  ((B - TW) * TF**3 + D**3 * TW) / 3 - A * Cz**2
    Izz = (B**3 * TF + (D - TF) * TW**3) / 12
    if ETABS:
        return {'P': 2*(B+D), 'C3': Cy, 'C2': D - Cz, 
            'A': A, 'AS3': 0.9*B*TF, 'AS2': (D-TF)*TW, 
            'I33': Iyy, 'I22': Izz}
    else:
        return {'P': 2*(B+D), 'Cy': Cy, 'Cz': D - Cz, 
            'A': A, 'Avy': 0.9*B*TF, 'Avz': (D-TF)*TW, 
            'Iyy': Iyy, 'Izz': Izz}


def R_props(D, B, ETABS=True):
    """Properties of rectangle"""
    A = B * D
    Iyy = B * D**3 / 12
    Izz = D * B**3 / 12
    if ETABS:
        return {'P': 2*(B+D), 'A': A, 'AS3': 0.833*A, 'AS2': 0.833*A, 
            'I33': Iyy, 'I22': Izz}
    else:
        return {'P': 2*(B+D), 'A': A, 'Avy': 0.833*A, 'Avz': 0.833*A, 
            'Iyy': Iyy, 'Izz': Izz}

    
def RH_props(D, B, TF, TW, ETABS=True):
    """Properties of hollow rectangle (tube)
    
    >>> RH_props(12, 15, 3, 2)
    {'P': 54, 'A': 108, 'Avy': 72, 'Avz': 60, 'Iyy': 2889.0, 'Izz': 1776.0}
    """
    A = B*D - (B-2*TW)*(D-2*TF)
    Iyy = B*D**3/12 - (B-2*TW)*(D-2*TF)**3/12
    Izz = D*B**3/12 - (D-2*TF)*(B-2*TW)**3/12
    if ETABS:
        return {'P': 2*(B+D), 'A': A, 'AS3': 2*TF*(B-2*TW), 'AS2': 2*TW*(D-2*TF), 
            'I33': Iyy, 'I22': Izz}
    else:
        return {'P': 2*(B+D), 'A': A, 'Avy': 2*TF*(B-2*TW), 'Avz': 2*TW*(D-2*TF), 
            'Iyy': Iyy, 'Izz': Izz}


def C_props(D, ETABS=True):
    """Properties of a circle or rod"""
    A = 0.25 * pi * D**2
    I = pi * D**4 / 64   # check this
    if ETABS:
        return  {'P': pi*D, 'A': A, 'AS3': 0.9*A, 'AS2': 0.9*A, 'I33': I, 'I22': I}
    else:
        return  {'P': pi*D, 'A': A, 'Avy': 0.9*A, 'Avz': 0.9*A, 'Iyy': I, 'Izz': I}


def CHS_props(D, T, ETABS=True):
    """Properties of a hollow circle or pipe"""
    A = 0.25 * pi * (D**2 - (D-2*T)**2)
    I = pi * (D**4 - (D-2*T)**4) / 64
    if ETABS:
        return  {'P': pi*D, 'A': A, 'AS3': 0.6*A, 'AS2': 0.6*A, 'I33': I, 'I22': I}
    else:
        return  {'P': pi*D, 'A': A, 'Avy': 0.6*A, 'Avz': 0.6*A, 'Iyy': I, 'Izz': I}


def I_props(D, B, TF, TW, ETABS=True):
    """Properties of an I-section
    >>> I_props(12, 15, 3, 2)
    {'P': 74, 'A': 90, 'Avy': 72, 'Avz': 30, 'Iyy': 2767.5, 'Izz': 858.0}
    """
    A = B*D - (B-TW)*(D-2*TF)
    Iyy = B*D**3/12 - (B-TW)*(D-2*TF)**3/12
    Zyy = 2 * Iyy / D
    Syy = B * TF * (D - TF) + 0.25 * TW * (D - 2 * TF)**2
    Izz = TF*B**3/6 - (D - 2*TF) * TW**3 / 12
    Zzz = 2 * Izz / B
    Szz = 0.5 * TF * B**2 + 0.25 * (D - 2*TF) * TW**2
    if ETABS:
        return {'P':2*D+4*B-2*TW ,'A': A, 'AS3': 5/3*TF*B, 'AS2': TW*(D-TF), 'I33': Iyy, 'I22': Izz,
        'S33': Zyy, 'S22': Zzz, 'Z33': Syy, 'Z22': Szz}
    else:
        return {'P':2*D+4*B-2*TW ,'A': A, 'Avy': 2*TF*B, 'Avz': TW*D, 'Iyy': Iyy, 'Izz': Izz,
        'Zyy': Zyy, 'Zzz': Zzz, 'Syy': Syy, 'Szz': Szz}


def CH_props(D, B, TF, TW, ETABS=True):
    """Properties of an C-section
    """
    A = B*D - (B-TW)*(D-2*TF)
    P = 2*D+4*B-2*TW
    #Cy = (D**2 * TW + TF**2 * (B0 - TW)) / A
    Cy = (TF*B**2 + 0.5*(D-2*TF)*TW**2) / A
    Cz = 0
    Iyy = B*D**3/12 - (B-TW)*(D-2*TF)**3/12
    Izz = (2*TF*B**3 + (D-2*TF)*TW**3) / 3 - A * Cy**2
    if ETABS:
        return {'P': P ,'A': A, 'AS3': 2*TF*B, 'AS2': TW*D, 'I33': Iyy, 'I22': Izz}
    else:
        return {'P': P ,'A': A, 'Avy': 2*TF*B, 'Avz': TW*D, 'Iyy': Iyy, 'Izz': Izz}


def deck_props(DD, DR, B, TT, TB): 
    """Properties of a trapezoidal deck when given deck depth (DD), rib depth (DR)
    rib spacing (B), rib width at top (TT), rib width at bottom (TB)
    - P is the length of the deck on the underside (topside is ignored)
    - A is the cross-sectional area of concrete per rib
    - D_AVE is the average depth of concrete
    
    NB At the moment no flexural properties are calculated"""
    A = DD * B + 0.5 * (TT + TB) * DR
    P = B - TT + TB + ((TT - TB)**2 + 4 * DR**2)**0.5
    return {'P': P, 'A': A, 'D_AVE': A / B}


def get_cat_sec_props(f_dict, section_def_dict):
    """Returns catalogue section properties as a dictionary when given 
    standard ETABS section information in the form of a dictionary"""
    # area, units = None, None
    prop_file = f_dict['FILE']
    #print('prop_file: ', prop_file)
    if prop_file in section_def_dict.keys():
        #print('shape: ', f_dict.get('SHAPE'))
        shape_dict = section_def_dict[prop_file]['SECTIONS'].get(f_dict.get('SHAPE'))
        #print('SHAPE_dict: ', shape_dict)
        #if shape_dict:
    return shape_dict # area, units
