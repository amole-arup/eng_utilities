""""""

from collections import namedtuple

from eng_utilities.section_utilities import *


## =============================
## === E2K Section Utilities ===
## =============================


Cardinal_Points_Dict = {
    1: 'Bottom left',
    2: 'Bottom center',
    3: 'Bottom right',
    4: 'Middle left',
    5: 'Middle center',
    6: 'Middle right',
    7: 'Top left',
    8: 'Top center',
    9: 'Top right',
    10: 'Centroid',
    11: 'Shear center',
}


def cardinal_points_offsets(num, D, B, CY=0, CZ=0):
    """It is assumed that the centroidal distances are measured 
    to the centre of the bounding box of the section - i.e. for 
    an L-section both will be positive"""
    B = D if B==0 else B # for circular sections
    return {
    1: (0.5*B-CY, 0.5*D-CZ), # 'Bottom left'
    2: (0, 0.5*D-CY), # 'Bottom center',
    3: (-0.5*B-CY, 0.5*D-CZ), # 'Bottom right',
    4: (0.5*B-CY, 0), # 'Middle left',
    5: (-CY, -CZ), # 'Middle center',
    6: (-0.5*B-CY, 0), # 'Middle right',
    7: (0.5*B-CY, -0.5*D-CZ), # 'Top left',
    8: (0, -0.5*D-CZ), # 'Top center',
    9: (-0.5*B-CY, -0.5*D-CZ), # 'Top right',
    10: (0, 0), # 'Centroid',
    11: (0, 0), # 'Shear center',
    }.get(num, (0, 0))


# The following functions all expect to receive a dictionary
# that defines a section and to return a dicitonary containing
# section properties
def CER_props_func(pdict):
    """Concrete Encased Rectangle"""
    return {}


def CEC_props_func(pdict):
    """Concrete Encased Circle"""
    return {}


def NP_props_func(pdict, np_dict):
    """Non-prismatic"""
    return {}


def SD_props_func(pdict, sd_dict):
    """Section Designer"""
    pass


def A_props_func(pdict):
    """Properties of single angle"""
    D = pdict.get('D')
    B = pdict.get('B')
    TF = pdict.get('TF')
    TW = pdict.get('TW')
    units = pdict.get('UNITS')
    GWA = f'STD A({units}) {D} {B} {TW} {TF}'
    if D and B and TF and TW:
        return {**A_props(D, B, TF, TW), **{'GWA': GWA}}
    else:
        return {}


def AA_props_func(pdict):
    """Properties of double angle
    Note that B is the sum of two legs + the distance between 
    the angles (DIS)
    """
    D = pdict.get('D')
    B = pdict.get('B')
    TF = pdict.get('TF')
    TW = pdict.get('TW')
    DIS = pdict.get('DIS', 0)
    units = pdict.get('UNITS')
    GWA = f'STD D({units}) {D} {0.5*(B-DIS)} {TW} {TF}'
    if D and B and TF and TW:
        return {**AA_props(D, B, TF, TW, DIS), **{'GWA': GWA}}
    else:
        return {}


def T_props_func(pdict):
    """Properties of a Tee
    """
    D = pdict.get('D')
    B = pdict.get('B')
    TF = pdict.get('TF')
    TW = pdict.get('TW')
    units = pdict.get('UNITS')
    GWA = f'STD T({units}) {D} {B} {TW} {TF}'
    if D and B and TF and TW:
        return {**T_props(D, B, TF, TW), **{'GWA': GWA}}
    else:
        return {}


def R_props_func(pdict):
    """Properties of rectangle (from dictionary containing 'D' & 'B')"""
    D = pdict.get('D')
    B = pdict.get('B')
    units = pdict.get('UNITS')
    GWA = f'STD R({units}) {D} {B}'
    if D and B:
        return {**R_props(D, B), **{'GWA': GWA}}
    else:
        return {}


def RH_props_func(pdict):
    """"Properties of hollow rectangle (tube), from dictionary containing 'D', 'B', TF' & 'TW'
    
    >>> RH_props_func({'D':15, 'B':12, 'TF':3, 'TW':2})
    {'P': 54, 'A': 108, 'Avy': 72, 'Avz': 60, 'Iyy': 2889.0, 'Izz': 1776.0}
    """
    D = pdict.get('D')
    B = pdict.get('B')
    TF = pdict.get('TF')
    TW = pdict.get('TW')
    units = pdict.get('UNITS')
    GWA = f'STD RHS({units}) {D} {B} {TW} {TF}'
    if D and B and TF and TW:
        return {**RH_props(D, B, TF, TW), **{'GWA': GWA}}
    else:
        return {}


def C_props_func(pdict):
    D = pdict.get('D') 
    units = pdict.get('UNITS')
    GWA = f'STD C({units}) {D}'
    if D:
        return {**C_props(D), **{'GWA': GWA}}
    else:
        return {}


def CHS_props_func(pdict):
    """Properties of a hollow circle or pipe (from dictionary containing 'D' & 'T')"""
    D = pdict.get('D') 
    T = pdict.get('T')
    units = pdict.get('UNITS')
    GWA = f'STD CHS({units}) {D} {T}'
    if D and T:
        return {**CHS_props(D, T), **{'GWA': GWA}}
    else:
        return {}


def General_props_func(pdict):
    """Properties of an I-section when given a dictionary of D, B, TF, TW

    This data comes from the 'FRAMESECTION' keyword. Typical data:  
    {'MATERIAL': "SM570",  'SHAPE': "General",  
    'D': 0.413, 'B': 0.413, 'AREA': 0.021, 'AS2': 0.021, 'AS3': 0.021, 
    'I33': 0.000354, 'I22': 0.000354, 'I23': 8.673617E-19, 
    'S33POS': 0.001716, 'S33NEG': 0.001716, 'S22POS': 0.001716, 'S22NEG': 0.001716, 
    'R33': 0.12991, 'R22': 0.12991, 'Z33': 0.001716, 'Z22': 0.001716, 'TORSION': 0.003433}

    The resulting GWA description is 'EXP(m) 0.021 0.000354 0.000354 0.003433 1.0 1.0'
    
    >>> General_props_func({'D':15, 'B':12, 'TF':3, 'TW':2})
    {'P': 74, 'A': 90, 'Avy': 72, 'Avz': 30, 'Iyy': 2767.5, 'Izz': 858.0}
    """
    AREA = pdict.get('AREA', 0)
    Iyy = pdict.get('I33', 0)
    Izz = pdict.get('I22', 0)
    J = pdict.get('TORSION', 0)
    Kyy = 1.0 if (not AREA) else pdict.get('AS3', 0) / AREA
    Kzz = 1.0 if (not AREA) else pdict.get('AS2', 0) / AREA
    units = pdict.get('UNITS')
    GWA = f'EXP({units}) {AREA} {Iyy} {Izz} {J} {Kyy} {Kzz}'
    if AREA:
        return {**pdict, **{'GWA': GWA}}
    else:
        return {} 


def I_props_func(pdict):
    """Properties of an I-section when given a dictionary of D, B, TF, TW
    
    >>> I_props_func({'D':15, 'B':12, 'TF':3, 'TW':2})
    {'P': 74, 'A': 90, 'Avy': 72, 'Avz': 30, 'Iyy': 2767.5, 'Izz': 858.0}
    """
    D = pdict.get('D')
    B = pdict.get('B')
    TF = pdict.get('TF')
    TW = pdict.get('TW')
    units = pdict.get('UNITS')
    GWA = f'STD I({units}) {D} {B} {TW} {TF}'
    if D and B and TF and TW:
        return {**I_props(D, B, TF, TW), **{'GWA': GWA}}
    else:
        return {} 


def CH_props_func(pdict):
    """Properties of an Channel-section when given a dictionary of D, B, TF, TW
    """
    D = pdict.get('D')
    B = pdict.get('B')
    TF = pdict.get('TF')
    TW = pdict.get('TW')
    units = pdict.get('UNITS')
    GWA = f'STD CH({units}) {D} {B} {TW} {TF}'
    if D and B and TF and TW:
        return {**CH_props(D, B, TF, TW), **{'GWA': GWA}}
    else:
        return {} 


def deck_props_func(pdict):
    """Properties of a trapezoidal deck when given a dictionary of DD, DR, B, TT, TB
    P is the length of the deck
    A is the cross-sectional area of concrete per rib
    D_AVE is the average depth of concrete
    NB At the moment no flexural properties are calculated"""
    DD = pdict.get('DECKSLABDEPTH')
    DR = pdict.get('DECKRIBDEPTH')
    B = pdict.get('DECKRIBSPACING')
    TT = pdict.get('DECKRIBWIDTHTOP')
    TB = pdict.get('DECKRIBWIDTHBOTTOM')
    TD = pdict.get('DECKSHEARTHICKNESS')
    
    if DD and DR and B and TT and TB:
        deck_props_dict = deck_props(DD, DR, B, TT, TB)
        P = deck_props_dict.get('P')
        deck_props_dict['T_AVE'] = TD * P / B
        return deck_props_dict
    else:
        return {}


def get_cat_sec_props(f_dict, section_def_dict):
    """Returns catalogue section properties as a dictionary when given 
    standard ETABS section information in the form of a dictionary"""
    area, units = None, None
    prop_file = f_dict['FILE']
    #print('prop_file: ', prop_file)
    if prop_file is None:
        shape_dict = {}
    if prop_file in section_def_dict.keys():
        #print('shape: ', f_dict.get('SHAPE'))
        # look up shape properties - if lookup fails return empty dictionary
        shape_dict = section_def_dict.get(prop_file, {}).get('SECTIONS',{}).get(f_dict.get('SHAPE'))
        #print('SHAPE_dict: ', shape_dict)
        #if shape_dict:
    return shape_dict # area, units
