"""Utilities for generating GWA text files from the processed
dictionary generated from ETABS text files (E2K, $ET)

TODO:
* Add assemblies for spandrels and piers
"""

import sqlite3
from os.path import exists
from dateutil.parser import parse

from eng_utilities.general_utilities import Units, is_numeric, try_numeric, units_conversion_factor
from eng_utilities.geometry_utilities import *
from eng_utilities.E2K_section_utilities import cardinal_points_offsets
from eng_utilities.polyline_utilities import build_force_line, interpolate_force_line3 #, add_force_line

### Units from general utilities
#Units = namedtuple('Units', 'force length temperature')


## ====================
## ===  GWA  Funcs  ===
## ====================

def check_GSA_ver(GSA_ver):
    GSA_list = str(GSA_ver).split('.')
    if is_numeric(GSA_list[0]):
        GSA_num = try_numeric(GSA_list[0])
        if len(GSA_list) > 1:
            if is_numeric(GSA_list[1]):
                GSA_num += try_numeric('0.' + GSA_list[1])
        return GSA_num
    return 10


def GWA_list_shrink(num_list):
    """Returns a string containing sorted abbreviated values
    e.g. [5, 8, 4, 3, 1] -> "1 3 to 5 8"
    Write to GWA file using ' '.join(GWA_sort(num_list))"""
    glist = sorted(num_list)
    glist.append(0)
    olist = []
    n0 = glist[0]
    n1 = glist[0]
    for n in glist[1:]:
        if n == n1 + 1:
            n1 = n
        else:
            olist.extend([str(n0), 'to', str(n1)] if (n0 != n1) else [str(n1)])
            n0 = n
            n1 = n    
    return ' '.join(olist)


GSA_SECT_SHAPE = {
    1: 'I-Section',
    2: 'Castellated I-Sections',
    3: 'Channels',
    4: 'T-section',
    5: 'Angles',
    6: 'Double Angles',
    7: 'CHS / Pipe',
    8: 'Round',
    9: 'SHS / RHS',
    10: 'Square',
    1033:'Ovals',
    1034: 'Double Channels',
}


def sectlib_to_dict(tk_list, filepath=None, df_ready=False):
    """
    """
    fp = filepath if filepath else r'C:\Program Files\Oasys\GSA 10.1\sectlib.db3'
    if not exists(fp):
        return {}
    
    conn = sqlite3.connect(fp)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    dicts = {}
    for table, key in tk_list:
        cur.execute(f"SELECT * From `{table}`")
        the_dict = {}
        rows = cur.fetchall()
        headers = [column[0] for column in cur.description]
        key_index = headers.index(key)
        for i, row in enumerate(rows):
            row_key = row[key_index]
            if df_ready:
                row_dict = {headers[j]:val for j, val in enumerate(row)}
                the_dict[i] = row_dict
            else:
                row_dict = {headers[j]:val for j, val in enumerate(row) if j != key_index}
                the_dict[row_key] = row_dict
        dicts[(table, key)] = the_dict
    conn.close()
    return dicts


def import_GSA_sections(filepath=None):
    """Collates all the section data from the Oasys sectlib database, which is 
    organised in four tables.
    
    # Catalogues     CatalogueData           Types           Sect
    # CAT_NUM (K) == CATDATA_CAT_NUM      == TYPE_CAT_NUM
    #                CATDATA_TYPE_NUM (K) == TYPE_NUM (K) == SECT_TYPE_NUM
    #                                        TYPE_SHAPE   == SECT_SHAPE
    #                                                        SECT_ID (K)

    The following code will insert the data into a Pandas DataFrame
    >> GSA_secs_df = pd.DataFrame.from_dict(combined_sect_dict(), orient='index')
    >> # The important columns are as follows:
    >> sec_cols = ['SECT_NAME', 'SECT_NUM', 'SECT_TYPE_NUM', 'SECT_SHAPE', 'SECT_SUPERSEDED', 
            'TYPE_NAME', 'TYPE_SHAPE', 'TYPE_ABR', 'TYPE_SECT_ABR', 'TYPE_SECT_FINISH'] + \
            ['CAT_ABR', 'CATDATA_CAT_NUM', 'CAT_NAME', 'SECT_DATE_ADDED']
    >> GSA_secs_df[sec_cols]
    """
    type_cols = ['TYPE_NUM', 'TYPE_NAME', 'TYPE_SHAPE', 'TYPE_ABR', 'TYPE_SECT_ABR', 'TYPE_SECT_FINISH']
    catdata_cols = ['CATDATA_TYPE_NUM', 'CATDATA_CAT_NUM']
    cat_cols = ['CAT_NUM', 'CAT_ABR', 'CAT_NAME']
    
    tk_list = (
        ('Catalogues', 'CAT_NUM'),
        ('CatalogueData', 'CATDATA_TYPE_NUM'),
        ('Types', 'TYPE_NUM'),
        ('Sect', 'SECT_ID')
        )
    dicts = sectlib_to_dict(tk_list, filepath)
    cat_dict, catdata_dict, type_dict, sect_dict  = [dicts[tk] for tk in tk_list]
        
    # for k, v in sect_dict.items():    
    for i, (k, v) in enumerate(sect_dict.items()):
        sect_type_num = v['SECT_TYPE_NUM']
        type_data = type_dict.get(sect_type_num, {})
        v.update({k0: v0 for k0, v0 in type_data.items() if k0 in type_cols})
        catdata_data = catdata_dict.get(sect_type_num, {})
        v.update({k0: v0 for k0, v0 in catdata_data.items() if k0 in catdata_cols})
        catdata_cat_num = v.get('CATDATA_CAT_NUM', -1)
        #catdata_cat_num = catdata_data.get('CATDATA_CAT_NUM')
        if i in (0,1,20,50,1000):
            print(catdata_cat_num)
        cat_data = cat_dict.get(catdata_cat_num, {})
        v.update({k0: v0 for k0, v0 in cat_data.items() if k0 in cat_cols})
        sect_date_added = v['SECT_DATE_ADDED']
        v['SECT_DATE_ADDED'] = parse(sect_date_added)
    
    return sect_dict


def GWA_GEO(pts_list, units=None):
    """"""
    x0, y0 = pts_list[0]
    # 'GEO P(mm) M(36361|-6881.57) L(37443.5|-6256.57) L(37443.5|-4456.57) L(36847.9|-3425) L(34986|-4500)'
    if units:
        return f'GEO P({units}) M({x0}|{y0}) ' + ' '.join([f'L({x}|{y})' for x, y in pts_list[1:]])
    else:
        return f'GEO P M({x0}|{y0}) ' + ' '.join([f'L({x}|{y})' for x, y in pts_list[1:]])


def GWA_sec_gen(pdict):
    """Returns mapped GWA section catalog strings for common section names,
    such as UC, UB, W. Others could easily be added. This only works with earlier
    GWA section definitions. In GSA 10, the section specification is more complex.
    
    Two development options are possible:
    1. Section dimensions could be identified in some cases, e.g. "HSS5X.250" 
        could be interpreted as a hollow section of diameter 5" and thickness 1/4"
    2. It would be possible to develop a more sophisticated approach that would 
       reference the section catalogues. This would require both catalogues to be available.
    """
    units = pdict.get('UNITS')
    
    sec_name = pdict['SHAPE'].replace('X','x')
    if sum(tt.isnumeric() for tt in sec_name[1:].split('x') if sec_name.startswith('W')) == 2:
        return ('CAT W ' if float(sec_name[1:].split('x')[0]) < 45 else 'CAT CA-W ') + sec_name 
    elif sum(tt.isnumeric() for tt in sec_name[2:].split('x') if sec_name.startswith('UB')) == 3:
        return 'CAT UB ' + sec_name
    elif sum(tt.isnumeric() for tt in sec_name[2:].split('x') if sec_name.startswith('UC')) == 3:
        return 'CAT UC ' + sec_name
    elif pdict.get('SECTION_TYPE') in ('PIPE', 'STEEL_PIPE'):
        D, T = [pdict.get(s) for s in ('OD', 'TDES')]
        if all((D, T)):
            return f'STD CHS({units}) {D} {T}'
        else:
            return 'EXPLICIT'
    elif pdict.get('SECTION_TYPE') in ('I_SECTION', 'STEEL_I_SECTION'):
        D, B, TW, TF = [pdict.get(s) for s in ('D', 'BF', 'TW', 'TF')]
        if all((D, B, TW, TF)):
            return f'STD I({units}) {D} {B} {TW} {TF}'
        else:
            return 'EXPLICIT'
    elif pdict.get('SECTION_TYPE') in ('TEE', 'STEEL_TEE'):
        D, B, TW, TF = [pdict.get(s) for s in ('D', 'BF', 'TW', 'TF')]
        if all((D, B, TW, TF)):
            return f'STD T({units}) {D} {B} {TW} {TF}'
        else:
            return 'EXPLICIT'
    elif pdict.get('SECTION_TYPE') in ('CHANNEL', 'STEEL_CHANNEL'):
        D, B, TW, TF = [pdict.get(s) for s in ('D', 'BF', 'TW', 'TF')]
        if all((D, B, TW, TF)):
            return f'STD CH({units}) {D} {B} {TW} {TF}'
        else:
            return 'EXPLICIT'
    elif pdict.get('SECTION_TYPE') in ('DOUBLE_CHANNEL', 'STEEL_DOUBLE_CHANNEL'):
        D, B, TW, TF, DIS = [pdict.get(s) for s in ('D', 'BF', 'TW', 'TF', 'DIS')]
        if all((D, B, TW, TF)):
            return f'STD CH({units}) {D} {B} {TW} {TF}'
        else:
            return 'EXPLICIT'
    elif pdict.get('SECTION_TYPE') in ('ANGLE', 'STEEL_ANGLE'):
        D, B, TW, TF = [pdict.get(s) for s in ('D', 'B', 'TW', 'TF')]
        if all((D, B, TW, TF)):
            return f'STD A({units}) {D} {B} {TW} {TF}'
        else:
            return 'EXPLICIT'
    elif pdict.get('SECTION_TYPE') in ('DOUBLE_ANGLE', 'STEEL_DOUBLE_ANGLE'):
        D, B, TW, TF, DIS = [pdict.get(s) for s in ('D', 'B', 'TW', 'TF', 'DIS')]
        if all((D, B, TW, TF)):
            return f'STD D({units}) {D} {0.5*(B-DIS)} {TW} {TF}'
        else:
            return 'EXPLICIT'
    elif pdict.get('SECTION_TYPE') in ('BOX', 'STEEL_BOX', 'STEEL_TUBE', 'TUBE'):
        D, B, TW, TF = [pdict.get(s) for s in ('HT', 'B', 'TW', 'TF')]
        if not D:
            D = B
        if all((D, B, TW, TF)):
            return f'STD RHS({units}) {D} {B} {TW} {TF}'
        else:
            return 'EXPLICIT'
    else:
        return 'EXPLICIT'


def set_restraints(rstr):
    restdict = {'UX': 'x', 'UY': 'y', 'UZ': 'z', 
               'RX': 'xx', 'RY': 'yy', 'RZ': 'zz', }
    return ''.join([restdict[r] for r in rstr.split()])


def set_releases(rel, n=1):
    if rel == 'PINNED':
        rel_1 = 'FFFRRR'
        rel_2 = 'FFFFRR'
        if n == 1:
            return ['FFFRRR\tFFFFRR']
    else:    
        rels = rel.split()
        rel_1 = ''.join(['R' if r in rels else 'F' for r in ('PI', 'V2I', 'V3I', 'TI', 'M2I', 'M3I')])
        rel_2 = ''.join(['R' if r in rels else 'F' for r in ('PJ', 'V2J', 'V3J', 'TJ', 'M2J', 'M3J')])
    if n == 1:
        return ['\t'.join([rel_1, rel_2])]
    else:
        rels_list = [rel_1] + (n-1)*2*['FFFFFF'] + [rel_2]
        return ['\t'.join([r1, r2]) for r1, r2 in zip(rels_list[::2], rels_list[1::2])]


def set_offsets(offsets, offset_sys = None, cy = 0, cz = 0):
    """Takes 6 offset values from ETABS and converts into GSA standard
    
    ('LENGTHOFFI', 'OFFSETYI', 'OFFSETZI', 
        'LENGTHOFFJ', 'OFFSETYJ', 'OFFSETZJ')
    off_x1, off_x2, off_y, off_z
    
    Note that we need a coordinate system to implement the OFFSETYI settings,
    so these have been ignored
    """
    off_x1, off_x2 = offsets[0], offsets[3]
    
    return (off_x1, off_x2, cy, cz)


GSA_mass_dict = {
    ('N', 'm'): 'kg', ('kN', 'm'): 't', ('MN', 'm'): 'kt',
    ('N', 'mm'): 't', ('kN', 'mm'): 'kt',
    ('kgf', 'mm'): 'kg', ('tonf', 'mm'): 't',
    ('kgf', 'cm'): 'kg', ('tonf', 'cm'): 't',
    ('kgf', 'm'): 'kg', ('tonf', 'm'): 't',
    ('lb', 'in'): 'lb', ('kip', 'in'): 'kip', 
    ('lb', 'ft'): 'lb', ('kip', 'ft'): 'kip',
}


def set_GSA_mass_units(force_units, length_units):
    return GSA_mass_dict.get((force_units, length_units))


GSA_pressure_dict = {
    ('N', 'm'): 'Pa', ('kN', 'm'): 'kPa', ('MN', 'm'): 'MPa',
    ('N', 'cm'): 'N/cm²', ('kN', 'cm'): 'kN/cm²',
    ('N', 'mm'): 'MPa', ('kN', 'mm'): 'GPa',
    ('kgf', 'm'): 'kgf/m²', ('tonf', 'm'): 'tonf/cm²',
    ('kgf', 'cm'): 'kgf/cm²', ('tonf', 'cm'): 'tonf/cm²',
    ('kgf', 'mm'): 'kgf/mm²', ('tonf', 'mm'): 'tonf/cm²',
    ('lb', 'in'): 'psi', ('kip', 'in'): 'ksi', 
    ('lb', 'ft'): 'psf', ('kip', 'ft'): 'ksf',
}


def set_GSA_pressure_units(force_units, length_units):
    return GSA_pressure_dict.get((force_units, length_units))


def get_GSA_local_axes(line, angle=0, vertical_angle_tolerance=1.0):
    """Returns the local axes based on GSA default definitions
    
    Note that GSA defaults are different from ETABS.
    Default tolerance for verticality is one degree. To set it to be the same as
    ETABS, set vertical_angle_tolerance = `asin(0.001)*180/pi` = 0.057296 deg (or
    simply make the value negative, and it will be calculated automatically).
    
    Args:
        line: a 2-tuple of 3-tuples representing the start and end nodes in 
            global cartesian coordinates
        angle (float): rotation in degrees as positive rotation about axis 1.
        vertical_angle_tolerance (float): angle tolerance in degrees
    
    Returns:
        a 3-tuple of 3-tuples representing the local unit coordinates (x,y,z) in 
            global cartesian coordinates
    
    >>> fmt_3x3(get_GSA_local_axes(((0,0,0),(0,0,3.5)),0),'7.4f')
    '( 0.0000,  0.0000,  1.0000), ( 0.0000,  1.0000,  0.0000), (-1.0000,  0.0000,  0.0000)'
    >>> fmt_3x3(get_GSA_local_axes(((0,0,0),(0,0,3.5)),30),'7.4f')
    '( 0.0000,  0.0000,  1.0000), (-0.5000,  0.8660,  0.0000), (-0.8660, -0.5000,  0.0000)'
    >>> fmt_3x3(get_GSA_local_axes(((8.5,0,3.5),(8.5,1,3.5)), 30),'7.4f')
    '( 0.0000,  1.0000,  0.0000), (-0.8660,  0.0000,  0.5000), ( 0.5000,  0.0000,  0.8660)'
    >>> fmt_3x3(get_GSA_local_axes(((0,0,0),(7.2,9.0,3.5)),30),'7.4f')
    '( 0.5977,  0.7472,  0.2906), (-0.7670,  0.4276,  0.4784), ( 0.2332, -0.5088,  0.8287)'
    """
    #vert = True
    vector = sub3D(line[1], line[0])
    dir1 = unit3D(vector)
    dir2 = (0,0,0)
    dir3 = (0,0,0)
    #length = mag3D(vector)
    ang_tol_rad = vertical_angle_tolerance * pi / 180 if (vertical_angle_tolerance >= 0) else asin(0.001)*180/pi
    ang_rad = angle * pi / 180
    
    if abs(asin(sin3D(vector, (0, 0, 1)))) < ang_tol_rad: # Column
        #vert = True
        dir1 = (0, 0, 1)
        dir2 = (0, 1, 0) if angle == 0 else (-sin(ang_rad), cos(ang_rad), 0)
        dir3 = (-1, 0, 0) if angle == 0 else (-cos(ang_rad), -sin(ang_rad), 0)
    else:  # Not a column
        #vert = False
        dir2_ = unit3D(cross3D((0,0,1), dir1))
        dir3_ = unit3D(cross3D(dir1, dir2_))
        dir2 = dir2_ if angle == 0 else rotQ(dir2_, ang_rad, dir1)
        dir3 = dir3_ if angle == 0 else rotQ(dir3_, ang_rad, dir1)
    return dir1, dir2, dir3


def write_GWA(E2K_dict, GWApath, GSA_ver=10, add_poly=False):
    """Generates a basic GWA file (GSA text file format) 
    from the ETABS dict generated by `run_all` in this module
    :param E2K_dict: The dictionary containing ETABS model data that is 
            generated by `run_all` in this module, containing the following dictionaries:
            ['Stories', 'Points', 'LineDict', 'Lines', 'Areas', 'Groups', 'LoadCases', 'LoadCombs']
    :param GWApath: The file name, including path, for the output GWA file
    :param GSA_ver: This should be provided as an integer or float (e.g. 10 or 10.1)
    :param add_poly: Whether to generate new membrane elements for each storey
    :return: GWA file for reading into GSA"""
    # ==================================
    # ============= Titles =============
    # ==================================
    # TITLE | title | sub-title | calc | job_no | initials
    title1_keys = E2K_dict.get('CONTROLS', {}).get('TITLE1',{}).keys() #'Title 1'
    title1 = list(title1_keys)[0] if title1_keys else ''
    title2_keys = E2K_dict.get('CONTROLS', {}).get('TITLE2',{}).keys() #'Title 2'
    title2 = list(title2_keys)[0] if title2_keys else ''
    
    GSA_num = check_GSA_ver(GSA_ver)
    # =================================
    # ============= Units =============
    # =================================
    
    units = E2K_dict['UNITS']
    print(f'Units are {units}')
    grav_dict = {'m': 9.80665, 'cm': 980.665, 'mm': 9806.65, 'in': 32.2, 'ft': 386.4}
    grav = grav_dict.get(units.length, 9.81)
    force_factor = units_conversion_factor(('N', units.force))
    length_factor = units_conversion_factor(('m', units.length))
    stress_factor = force_factor / length_factor**2
    stress_units = set_GSA_pressure_units(units.force, units.length)
    mass_factor = force_factor / length_factor
    mass_units = set_GSA_mass_units(units.force, units.length)
    #print('F units:', units.force, force_factor, ', L units:', units.length, length_factor)
    #print('S units:', stress_units, stress_factor, ', M units:', mass_units, mass_factor)
    #print('T units:', "\xb0" + units.temperature, 1.0, '\n')
    if units.force not in ('N', 'kN', 'MN', 'lb', 'ton'):
        print(f'**Non-standard units**: \nCheck derived unit and factors based on {units}:')
        print(f'\tStress units: {stress_units}, {stress_factor}\n\tMass units: {mass_units}, {mass_factor}')
    
    MAT_PROPS_dict, FRAME_PROPS_dict, SHELL_PROPS_dict, SPRING_PROPS_dict = [
        E2K_dict.get(k1,{}).get(k2,{}) for k1, k2 in (
        ('MATERIAL PROPERTIES', 'MATERIAL'), 
        ('FRAME SECTIONS', 'FRAMESECTION'), 
        ('SHELL PROPERTIES', 'SHELLPROP'),
        ('POINT SPRING PROPERTIES', 'POINTSPRING'),
    )]

    STORY_dict, DIAPHRAGMS_dict, DIAPHRAGM_GROUPS_dict, DIAPHRAGM_LOOPS_dict = [
        E2K_dict.get(k1,{}).get(k2,{}) for k1, k2 in (
        ('STORIES - IN SEQUENCE FROM TOP', 'STORY'),
        ('DIAPHRAGM NAMES', 'DIAPHRAGM'),
        ('DIAPHRAGM NAMES', 'GROUPS'),
        ('DIAPHRAGM NAMES', 'LOOPS'))]
    
    NODE_dict, LINE_dict, AREA_dict = [E2K_dict.get(k1,{}).get(k2,{}) for k1, k2 in (
        ('POINT ASSIGNS', 'POINTASSIGN'), 
        ('LINE ASSIGNS', 'LINEASSIGN'), 
        ('AREA ASSIGNS', 'AREAASSIGN')
    )]
    
    GROUPS_dict = E2K_dict.get('GROUPS',{}).get('GROUP',{})
    
    if E2K_dict.get('LOAD PATTERNS'): # New ETABS after version 9.7
        LOADCASE_dict, WIND_dict, SEISMIC_dict, POINT_LOAD_dict, LINE_LOAD_dict, AREA_LOAD_dict = \
            [E2K_dict.get(k1,{}).get(k2,{}) for k1, k2 in (
            ('LOAD PATTERNS', 'LOADPATTERN'), 
            ('LOAD PATTERNS', 'WIND'),
            ('LOAD PATTERNS', 'SEISMIC'),
            ('POINT OBJECT LOADS', 'POINTLOAD'),
            ('FRAME OBJECT LOADS', 'LINELOAD'),
            ('SHELL OBJECT LOADS', 'AREALOAD')
        )]
    else:
        LOADCASE_dict, WIND_dict, SEISMIC_dict, POINT_LOAD_dict, LINE_LOAD_dict, AREA_LOAD_dict = \
            [E2K_dict.get(k1,{}).get(k2,{}) for k1, k2 in (
            ('STATIC LOADS', 'LOADCASE'), 
            ('STATIC LOADS', 'WIND'),
            ('STATIC LOADS', 'SEISMIC'),
            ('POINT OBJECT LOADS', 'POINTLOAD'),
            ('LINE OBJECT LOADS', 'LINELOAD'),
            ('AREA OBJECT LOADS', 'AREALOAD')
    )]

    LOAD_COMBO_dict = E2K_dict.get('LOAD COMBINATIONS',{}).get('COMBO',{})

    
    with open(GWApath, 'w') as gwa:
        # ** Writing initial lines to GWA **
        # NB "\xb0", "\xb2", "\xb3" are degree sign, squared and cubed symbols
        gwa.write(r'! This file was originally written by a Python script by the E2KtoJSON_code library' + '\n')
        gwa.write(r'! The model data was extracted from an ETABS text file (E2K or $ET)' + '\n')
        gwa.write(r'!  ' + E2K_dict.get('File', {}).get('Header','') + '\n')
        gwa.write(r'!' + '\n')
        gwa.write(r'! Notes:' + '\n')
        gwa.write(r'!   The user is warned to check the model and NOT to rely on the conversion' + '\n')
        gwa.write(r'!   (it is particularly important to check the units, the properties' + '\n')
        gwa.write(r'!    and the global forces).' + '\n')
        gwa.write(r'!' + '\n')
        #   TITLE | title | sub-title | calc | job_no | initials
        gwa.write('\t'.join(['TITLE.1', str(title1), str(title2), '', '', '', '', '\n']))
        
        gwa.write('\t'.join(['UNIT_DATA.1', 'FORCE', units.force, 
                             str(force_factor)]) + '\n')
        gwa.write('\t'.join(['UNIT_DATA.1', 'LENGTH', units.length, 
                             str(length_factor)]) + '\n')
        gwa.write('\t'.join(['UNIT_DATA.1', 'DISP', units.length, 
                             str(length_factor)]) + '\n')
        gwa.write('\t'.join(['UNIT_DATA.1', 'SECTION', units.length, 
                             str(length_factor)]) + '\n')
        gwa.write('\t'.join(['UNIT_DATA.1', 'STRESS', stress_units, 
                             str(stress_factor)]) + '\n') 
        # UNIT_DATA.1	TEMP	¢XÆC	1
        # Note that GSA uses a degree symbol that is missing at present...
        gwa.write('\t'.join(['UNIT_DATA.1', 'TEMP', "\xb0" + units.temperature, 
                             '1']) + '\n') 
        gwa.write('\t'.join(['UNIT_DATA.1', 'MASS', mass_units, 
                             str(mass_factor)]) + '\n') 
        gwa.write('\t'.join(['UNIT_DATA.1', 'ACCEL', units.length + r'/s' + "\xb2", 
                            str(length_factor)]) + '\n')
        gwa.write('\t'.join(['UNIT_DATA.1', 'TIME', 's', '1']) + '\n')
        
        
        # ============================
        # =====  Mat Properties  =====
        # ============================
        # ** Creating Materials Database and writing to GWA ** 
        # if ETABS_dict.get('MatProps'):
        # MAT_STEEL.3 | num | <mat> | fy | fu | eps_p | Eh
        # MAT_ANAL | num | MAT_ELAS_ISO | name | colour | 6 | E | nu | rho | alpha | G | damp |
        for m_name, m_dict in MAT_PROPS_dict.items():
            m_ID = m_dict.get('ID', '')
            E_val = m_dict.get('E', 0)
            nu = m_dict.get('U', 0)
            rho_w = m_dict.get('W', 0) if m_dict.get('W') else m_dict.get('WEIGHTPERVOLUME', 0)
            rho_m = rho_w / grav
            alpha = m_dict.get('A', 0)
            mat_type = m_dict.get('TYPE')
            ostr = [str(val) for val in ['MAT_ANAL', m_ID, 'MAT_ELAS_ISO', m_name, 
                    'NO_RGB', '6', E_val, nu, rho_m, alpha]]
            #gwa.write('PROP_SEC.2\t{:d}\t{:s}\t\t{:s}\t{:s}\n'.format(fp, name, '', desc))
            gwa.write('\t'.join(ostr) + '\n')
        
        
        # ============================
        # ===== Frame Properties =====
        # ============================
        # SECTION.2 | ref | name | memb:EXPLICIT | mat
        # PROP_SEC.1 | num | name | colour | anal | desc | prin | type | cost |
        # PROP_SEC.2 | num | name | colour | type | desc | anal | mat | grade |
        # PROP_SEC.3 | num | name | colour | mat | grade | anal | desc | cost | ref_point | off_y | off_z |
        # SECTION_MOD | ref | name | mod | centroid | stress | opArea | area | prin 
        #   | opIyy | Iyy | opIzz | Izz | opJ | J | opKy | ky | opKz | kz | opVol | vol | mass
        # ** Creating Properties Database and writing to GWA **
        mat_types = ['GENERIC', 'STEEL', 'CONCRETE', 'ALUMINIUM', 'GLASS', 'FRP', 'TIMBER']
        for fp_name, fp_dict in FRAME_PROPS_dict.items():
            mat_name = fp_dict.get('MATERIAL')
            m_dict = MAT_PROPS_dict.get(mat_name, {})
            mat_ID = m_dict.get('ID', 1)
            fp_ID = fp_dict.get('ID', '')
            desc = fp_dict.get('GWA', 'EXPLICIT')
            mat_type = m_dict.get('TYPE','').upper()
            mat_type = mat_type if (mat_type in mat_types) else 'GENERIC'
            ostr2 = ['PROP_SEC.2', str(fp_ID), fp_name, 'NO_RGB', '', 
                     desc, str(mat_ID), mat_type]
            #gwa.write('PROP_SEC.2\t{:d}\t{:s}\t\t{:s}\t{:s}\n'.format(fp, name, '', desc))
            gwa.write('\t'.join(ostr2) + '\n')
        
        
        # ============================
        # ===== Shell Properties =====
        # ============================
        
        # Area properties
        # PROP_2D.1 | num | name | axis | mat | type | thick | mass | bending
        for sp_name, sp_dict in SHELL_PROPS_dict.items():
            sp_ID = sp_dict.get('ID', '')
            desc = sp_dict.get('GWA', 'EXPLICIT')
            thickness = max([sp_dict.get(t, 0) for t in ['SLABTHICKNESS', 'WALLTHICKNESS','DECKTHICKNESS']])
            ostr = ['PROP_2D.2',str(sp_ID), sp_name,'NO_RGB','LOCAL','1',
                    'SHELL',str(thickness),'0.0','100.0%','100.0%','100.0%']
            # '.format(fa, name, 'LOCAL', 1, thickness))
            #gwa.write('PROP_SEC.2\t{:d}\t{:s}\t\t{:s}\t{:s}\n'.format(fp, name, '', desc))
            gwa.write('\t'.join(ostr) + '\n')
                
        
        # =============================
        # ===== Spring Properties =====
        # =============================
        # Spring properties
        # PROP_SPR.2 | num | name | colour | axis | type | curve_x | stiff_x 
        #            | curve_y | stiff_y | curve_z | stiff_z | damping
        # PROP_SPR.3 | num | name | colour | axis | SPRING | curve_x | stiff_x | curve_y | stiff_y | curve_z | stiff_z 
        #            | curve_xx | stiff_xx | curve_yy | stiff_yy | curve_zz | stiff_zz | damping
        # PROP_SPR.4 | num | name | colour | SPRING | curve_x | stiff_x | curve_y | stiff_y | curve_z | stiff_z 
        #            | curve_xx | stiff_xx | curve_yy | stiff_yy | curve_zz | stiff_zz | damping
        # PROP_SPR.4 | 2 | Gen_Spring | NO_RGB | GENERAL | 0 | 1990000000 | 0 
        #            | 2220000000 | 0 | 5.550000128e+10 | 0 | 0 | 0 
        #            | 0 | 0 | 4.440000102e+11 | 0
        for i, (spr_name, spr_dict) in enumerate(SPRING_PROPS_dict.items()):
            if not spr_dict.get('ID'):
                spr_dict['ID'] = i+1
            spr_ID = spr_dict.get('ID', 1) 
            k = [[0, spr_dict.get(dir, 0)] for dir in ('UX', 'UY', 'UZ', 'RX', 'RY', 'RZ')]
            
            ostr = ['PROP_SPR.4', str(spr_ID), spr_name, 'NO_RGB', 'SPRING'] + \
                   sum(k,[])
            gwa.write('\t'.join([str(n) for n in ostr]) + '\n')


        # =======================
        # =====    Nodes    =====
        # =======================
        # POINTASSIGN  "3895"  "BASE"  RESTRAINT "UX UY"  SPRINGPROP "PSpr1"  DIAPH "S1FLEX"
        # NODE.2 | num | name | colour | x | y | z | is_grid { | grid_plane | datum | grid_line_a | grid_line_b } | axis | is_rest { | rx | ry | rz | rxx | ryy | rzz } | is_stiff { | Kx | Ky | Kz | Kxx | Kyy | Kzz }
        # NODE.2    46      NO_RGB  -20.95  3.377499    33.3    NO_GRID 0   REST    1   1   1   0   0   0
        # NODE.3 | num | name | colour | x | y | z | restraint | axis |
        #                   mesh_size | springProperty | massProperty | damperProperty
        # NODE.3	2	name	NO_RGB	8.5	2.3	0.1	xyzxxyy	GLOBAL	0	1
        for n_name, n_dict in NODE_dict.items():
            nid = n_dict.get('ID', '')
            restr = n_dict.get('RESTRAINT', None)
            fixity = [set_restraints(restr)] if restr else []
            mesh_size = 0
            spr_prop = n_dict.get('SPRINGPROP')
            spr_ID = [SPRING_PROPS_dict.get(spr_prop,{}).get('ID')] if spr_prop else []
            more = []
            if spr_ID:
                more = spr_ID + more
            if more:
                more = [mesh_size] + more
            if more or fixity:
                more = fixity + ['GLOBAL'] + more
            coords = [str(w) for w in n_dict.get('COORDS', ('', '', ''))]
            ostr = ['NODE.3', nid, n_name, 'NO_RGB'] + \
                coords + more
            gwa.write('\t'.join([str(n) for n in ostr]) + '\n')


        # =======================
        # =====    Beams    =====
        # =======================
        # EL.2 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle 
        #      | is_rls { | rls { | k } } is_offset { | ox | oy | oz } | dummy
        # EL.4 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle 
        #      | is_rls { | rls { | k } } off_x1 | off_x2 | off_y | off_z | dummy | parent
        # EL.4  1       NO_RGB  BEAM    1   1   1   2   0   0   NO_RLS  0   0   0   0
        # NO_RLS | RLS | STIFF  - F, R, K
        
        # ** Writing Beams to GWA **
        bm_ID_max = max(bm_dict.get('ID',0) for bm_dict in LINE_dict.values())
        sh_ID_max = max(sh_dict.get('ID',0) for sh_dict in AREA_dict.values())
        next_ID = bm_ID_max + sh_ID_max + 1
        
        bm_max = 1
        for bm_name, bm_dict in LINE_dict.items():
            bm_ID = bm_dict.get('ID', '')
            N1 = bm_dict.get('N1', '')
            N2 = bm_dict.get('N2', '')
            bm_angle = bm_dict.get('ANG', 0)
            frame_prop_name = bm_dict.get('SECTION', None)
            fp_dict = FRAME_PROPS_dict.get(frame_prop_name, {})
            prop_num = fp_dict.get('ID', 1)
            
            # Intermediate nodes
            int_nodes = bm_dict.get('INTERMEDIATE_NODES', [])
            segment_num = len(int_nodes) + 1
            
            if segment_num > 1:
                new_ids = [next_ID + n for n in range(segment_num - 1)]
                next_ID = max(new_ids) + 1
                nds = [N1] + [nd[2] for nd in int_nodes] + [N2]
                node_pairs = [(n1, n2) for n1, n2 in zip(nds[:-1], nds[1:])]
                bm_ids = [bm_ID] + new_ids
                bm_dict['ELEMENT_IDS'] = tuple(bm_ids)
            else:
                node_pairs = [(N1, N2)]
                bm_ids = [bm_ID]

            # Offsets
            # rigid_zone = bm_dict.get('RIGIDZONE', 0) # not implemented
            offset_sys = bm_dict.get('OFFSETSYS', None)
            
            offsets = [bm_dict.get(offset, 0) for offset in 
                       ('LENGTHOFFI', 'OFFSETYI', 'OFFSETYI', 
                        'LENGTHOFFJ', 'OFFSETYJ', 'OFFSETYJ')]
            
            cardinal_point = bm_dict.get('CARDINALPT', 0)
            if cardinal_point:
                length_unit = fp_dict.get('UNITS')
                u_conv = units_conversion_factor((length_unit, units.length)) if length_unit else 1.0
                D, B, CY, CZ = [u_conv * fp_dict.get(dim, 0) for dim in ('D','B', 'C3', 'C2')]
                off_y, off_z = cardinal_points_offsets(cardinal_point, D, B, CY=0, CZ=0)
            else:
                off_y, off_z = 0, 0
            
            if any(offsets):
                 # NB OFFSETYI etc and offset_sys are not implemented
                offsets = set_offsets(offsets, offset_sys, off_y, off_z)
            else:
                offsets = (0, 0, off_y, off_z)
            offsets_txt = '\t'.join([str(n) for n in offsets])
            
            # Releases PINNED = 'FFFRRR\tFFFFRR'
            release = bm_dict.get('RELEASE', '')   # "TI M2I M3I TJ M2J M3J"
            
            if release:
                releases_txt = ['RLS\t' + rel for rel in set_releases(release, segment_num)]
            else:
                releases_txt = ['NO_RLS'] * segment_num
            
            # GSA cannot do property modifications for individual members
            #propmods = [bm_dict.get(propmod, 1) for propmod in 
            #           ('PROPMODA', 'PROPMODA2', 'PROPMODA3', 
            #            'PROPMODT', 'PROPMODI22', 'PROPMODI33')]
            # bm_dict.get('PROPMODM', 1)
            # bm_dict.get('PROPMODW', 1)
                      
            
            #ostr2 = [str(val) for val in ['EL.2', str(bm_ID), bm_name, 'NO_RGB', 'BEAM', prop_num, prop_num, 
            #        N1, N2, '', bm_angle, ]]
            #gwa.write('EL.2\t{:d}\t{:s}\t\t{:s}\t{:d}\t{:d}\t{}\t{}\t\t{:f}\n'.format(b + 1, bm_name, 'BEAM', prop_num, prop_num, ndict.get(pt_1), ndict.get(pt_2),bm_angle))
            #gwa.write('\t'.join(ostr2) + '\n')
            for beam_ID, (n_i, n_j), rel_txt in zip(bm_ids, node_pairs, releases_txt):
                dummy = '\tDUMMY' if prop_num == 1 else ''
                ostr4 = [str(val) for val in ['EL.4', str(beam_ID), bm_name, 'NO_RGB', 'BEAM', 
                        prop_num, prop_num, n_i, n_j, '', bm_angle]]
                if n_i != n_j:  # TODO: identify and eliminate the generation of these extra nodes
                    gwa.write('\t'.join(ostr4 + [rel_txt] + [offsets_txt]) + dummy + '\n')
            bm_max = max(bm_max, bm_dict.get('ID', 0))
        print('Max beam ID is:', bm_max)
        
        
        # ========================
        # =====    Shells    =====
        # ========================
        # EL.2 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle | is_rls { | rls { | k } } is_offset { | ox | oy | oz } | dummy
        # EL.4  1       NO_RGB  BEAM    1   1   1   2   0   0   NO_RLS  0   0   0   0
        # ** Writing Beams to GWA **
        memb_i = 1
        sh_max = bm_max
        for sh_name, sh_dict in AREA_dict.items():
            sh_ID = sh_dict.get('ID', '')
            num_pts = sh_dict.get('NumPts')
            nodes = [sh_dict.get('N'+str(n+1), '') for n in range(num_pts)]
            nodes_string = '\t'.join([str(n) for n in nodes])
            #sh_angle = sh_dict.get('ANG', 0)
            
            # These should be added to assemblies
            #pier = sh_dict.get('PIER', None)
            #spandrel = sh_dict.get('SPANDREL', None)
            #diaphragm = sh_dict.get('DIAPH', None)
            
            shell_prop_name = sh_dict.get('SECTION', None)
            sp_dict = SHELL_PROPS_dict.get(shell_prop_name, {})
            prop_num = sp_dict.get('ID', 1)
            if num_pts == 3:
                #gwa.write('EL.2\t{:d}\t{:s}\t\t{:s}\t{:d}\t{:d}\t{}\n'.format(
                #    bmax + a + 1, area_name, 'TRI3', prop_num, prop_num, pts_txt))
                ostr = ['EL.2', bm_max + sh_ID, sh_name, 'NO_RGB', 'TRI3', 
                    prop_num, prop_num, nodes_string]
                gwa.write('\t'.join([str(val) for val in ostr]) + '\n')
            elif num_pts == 4:
                ostr = ['EL.2', bm_max + sh_ID, sh_name, 'NO_RGB', 'QUAD4', 
                    prop_num, prop_num, nodes_string]
                gwa.write('\t'.join([str(val) for val in ostr]) + '\n')
            else:
                # Create polygonal members for n-gon with n > 4 
                el_type = 'SLAB'
                el_name, story_name = sh_name
                pname = f'Shell {el_name} (ID: {sh_ID}) @ {story_name}'
                # MEMB.8	1	fred	NO_RGB	SLAB	ALL	1	1	node_list	0	0	0	YES
                ostr = [str(val) for val in ['MEMB.8', memb_i , pname, 'NO_RGB', el_type, 'ALL', prop_num, 1, nodes_string.replace('\t', ' ')]]
                gwa.write('\t'.join(ostr) + '\n')
                memb_i += 1
            sh_max = bm_max + sh_ID
        memb_max = memb_i
        print('Max shell_ID is:', sh_max)
        
        
        # =======================
        # =====   Storeys   =====
        # =======================
        
        # ** Writing Storey Grid Planes to GWA **
        # GSA 10   GRID_PLANE.4  1   SPIRE-11    STOREY  0   247.967 0   0
        # GSA 8.7  GRID_PLANE.1   1   Ground floor    0   ELEV    0.000000    all     ONE 0.000000    0   0.0100000   STOREY  PLANE_CORNER    0.000000    0.000000
        # GSA 8.7  GRID_PLANE.1\t{num}\t{name}\t{axis}\tELEV\t{elev}\tall\tONE\t0.0\t0\t.01\tSTOREY\tPLANE_CORNER\t0.0\t0.0\n
        
        for s, (story, data) in enumerate(STORY_dict.items()):
            abs_elev = data.get('ABS_ELEV')
            if GSA_num < 10:
                axis = 0
                ostr = ['GRID_PLANE', str(s+1), story, str(axis), 'ELEV', str(abs_elev), 
                        'all\tONE\t0.0\t0\t.01\tSTOREY\tPLANE_CORNER\t0.0\t0.0']
                gwa.write('\t'.join(ostr) + '\n')
            else:
                ostr = ['GRID_PLANE.4', str(s+1), story, 'STOREY\t0',str(abs_elev),'0\t0']
                gwa.write('\t'.join(ostr) + '\n')
            
        
        # ======================
        # =====   Groups   =====
        # ======================
        
        # ** Writing Groups / Lists to GWA **
        # NB ETABS can have mixed groups, but GSA separates them by object type - e.g. beam lists, node lists etc 
        # ['POINT' 'LINE' 'AREA']
        list_id = 1
        if GROUPS_dict: # in case there are no Groups in the file
            for g_name, g_data in GROUPS_dict.items():
                n_list = []   
                el_list = []
                if g_data.get('POINT'):
                    for pt in g_data.get('POINT', []):
                        n_ID = NODE_dict.get(pt,{}).get('ID')
                        if n_ID:
                            n_list.append(n_ID)
                if g_data.get('LINE'):
                    for line in g_data.get('LINE', []):
                        bm_ID = LINE_dict.get(line,{}).get('ID')
                        if bm_ID:
                            el_list.append(bm_ID)
                if g_data.get('AREA'):
                    for area in g_data.get('AREA', []):
                        sh_ID = AREA_dict.get(area,{}).get('ID')
                        if sh_ID:
                            el_list.append(bm_max + sh_ID)
                if len(n_list) > 0:
                    # LIST | num | name | type | list
                    gwa.write('\t'.join(['LIST.1', str(list_id), g_name, 'NODE']))
                    gwa.write('\t')
                    gwa.write(GWA_list_shrink(n_list))
                    gwa.write('\n')
                    list_id += 1
                if len(el_list) > 0:
                    gwa.write('\t'.join(['LIST.1', str(list_id), g_name, 'ELEMENT']))
                    gwa.write('\t')
                    gwa.write(GWA_list_shrink(el_list))
                    gwa.write('\n')
                    list_id += 1

        # ========================
        # =====  Diaphragms  =====
        # ========================
        # RIGID.3 | name | primary_node | type | constrained_nodes | stage | parent_member
        # RIGID.3 <insert example>
        #
        # type: ALL, XY_PLANE, PIN, XY_PLANE_PIN, <link>
        
        # ** Writing Diaphragm Constraints to GWA **
        # Add diaphragms based on groups for each floor 
        for diaph_key, diaph_node_list in DIAPHRAGM_GROUPS_dict.items():
            n_list = [NODE_dict.get(nd_key,{}).get('ID') for nd_key in diaph_node_list]
            g_name = f'Diaphragm {diaph_key[1]} @ {diaph_key[0]}'
            # LIST | num | name | type | list
            gwa.write('\t'.join(['LIST.1', str(list_id), g_name, 'NODE']))
            gwa.write('\t')
            gwa.write(GWA_list_shrink([n for n in n_list if n is not None]))
            gwa.write('\n')
            list_id += 1

            diaph_dict = DIAPHRAGMS_dict.get(diaph_key[1], {})
            diaphragm_type = diaph_dict.get('TYPE', 'RIGID')
            g_string = r'"""' + str(g_name) + r'"""'
            if diaphragm_type == 'RIGID':
                d_type = 'XY_PLANE_PIN'
                ostr = [str(val) for val in ['RIGID.3', g_name, 0, d_type, g_string,'all','']]
                gwa.write('\t'.join(ostr) + '\n')

        
        # =================================
        # =====  Diaphragm Polylines  =====
        # =================================
        # POLYLINE | num | name | colour | grid_plane | num_dim | desc
        
        # MEMB.8 | num | name | colour | type (2D) | exposure | prop | group | topology | node | angle | mesh_size | is_intersector 
        #  | analysis_type | fire | limiting_temperature | time[4] | dummy 
        #  | off_z | off_auto_internal | reinforcement2d |
        # MEMB.8	1	fred	NO_RGB	SLAB	ALL	1	1	node_list	0	0	0	YES	
        # 	LINEAR	0	0	0	0	0	0	ACTIVE	0	NO	REBAR_2D.1	0.03	0.03	0
        
        if add_poly:
            print('Generating storey extents as a member')
            el_max = len(LINE_dict) + len(AREA_dict) + 10
            p_max = len(SHELL_PROPS_dict) + 1
            
            #pline_i = 1
            perim_i = memb_max + 1
            
            # for story_name, sdict in STORY_dict.items():
            for story_name in STORY_dict.keys():
                
                # Generate slab "member" to represent the perimeter of a beam mesh
                loop_list = [[NODE_dict[node]['ID'] for node in loop] 
                        for loop in DIAPHRAGM_LOOPS_dict.get(story_name, [])]
                #print(perimeter)
                loop_strings = [' '.join([str(n) for n in loop if n is not None]) for loop in loop_list]
                #print(f'loop_string ({story_name}): ', loop_strings)
            
                for j, loop_string in enumerate(loop_strings):
                    pname = f'Perimeter {j+1} @ {story_name}'
                    # MEMB.8	1	fred	NO_RGB	SLAB	ALL	1	1	node_list	0	0	0	YES
                    ostr = [str(val) for val in ['MEMB.8', el_max + perim_i , pname, 'NO_RGB', 'SLAB', 'ALL', p_max, 1, loop_string]]
                    gwa.write('\t'.join(ostr) + '\n')
                    perim_i += 1
                
                # Alternative representation using polylines
                """polylines = [[NODE_dict[node]['COORDS'] for node in loop] 
                        for loop in DIAPHRAGM_LOOPS_dict.get(story_name, [])]
                #print(story_name, )
            
                for j, polyline in enumerate(polylines):
                    pname = f'Perimeter {j+1} @ {story_name}'
                    pline_string = ''.join([str(p) for p in polyline]) #.replace(' ','')
                    ostr = [str(val) for val in ['POLYLINE', pline_i, pname, 'NO_RGB', pline_i, 3, pline_string]]
                    gwa.write('\t'.join(ostr) + '\n')
                    pline_i += 1"""
            
            



        # ========================
        # =====  Load Cases  =====
        # ========================
        
        # ** Writing Load Cases to GWA **
        # LOAD_TITLE.1 | case | title | type.1 | bridge

        # LOAD_GRAVITY.3 | name | elemlist | nodelist | case | x | y | z
        # LOADCASE_dict, WIND_dict, SEISMIC_dict
        
        load_type_dict = {
            'DEAD': 'DEAD',
            'SUPERDEAD': 'DEAD',
            'LIVE': 'IMPOSED',
            'WIND': 'WIND',
            'QUAKE': 'SEISMIC',
            'Dead': 'DEAD',
            'Live': 'IMPOSED',
            'Wind': 'WIND',
            'Seismic': 'SEISMIC',
        }

        for lc_name, lc_dict in LOADCASE_dict.items():
            lc_ID = lc_dict.get('ID', 0)
            lc_type = lc_dict.get('TYPE', '')
            ostr = [str(val) for val in ['LOAD_TITLE.1', lc_ID, lc_name, 
                                        load_type_dict.get(lc_type, 'UNDEF'),'']]
            gwa.write('\t'.join(ostr) + '\n')
            if lc_dict.get('SELFWEIGHT', None):
                gravity_factor = -1 * lc_dict.get('SELFWEIGHT', 0)
                ostr = [str(val) for val in ['LOAD_GRAVITY.3', '', 'all', 'all', lc_ID, 
                                            0, 0, gravity_factor]]
                gwa.write('\t'.join(ostr) + '\n')        
        

        # ========================
        # =====  Beam Loads  =====
        # ========================
        
        #   LINELOAD  "B2"  "L3"  TYPE "TRAPF"  DIR "GRAV"  LC "SDL"  FSTART 4.55  FEND 4.55  RDSTART 0  RDEND 0.3780311  
        #   LINELOAD  "B2"  "L3"  TYPE "TRAPF"  DIR "GRAV"  LC "SDL"  FSTART 4.55  FEND 4.55  RDSTART 0.3780311  RDEND 1  
        
        # LOAD_BEAM_PATCH.2		1744	2	GLOBAL	NO	Z	0	-4.55	-0.3780311	-4.55
        # LOAD_BEAM_PATCH.2		1744	2	GLOBAL	NO	Z	-0.3780311	-4.55	-1	-4.55
        
        # LOAD_BEAM_UDL.2 | name | list | case | axis | proj | dir | value
        # LOAD_BEAM_UDL.2	Uniform Load	286 287 295 296	2	GLOBAL	NO	Z	-5000
        # pos is negative if relative distance
        # LOAD_BEAM_PATCH.2 | name | list | case | axis | proj | dir | pos_1 | value_1 | pos_2 | value_2


        LC_ID_lookup_dict = {v['ID']: k for k, v in LOADCASE_dict.items()}
        print(LC_ID_lookup_dict)
        print(LOADCASE_dict)

        print('len of LINE_LOAD_dict', len(LINE_LOAD_dict))
        for load_key, load_list in LINE_LOAD_dict.items():
            member, story, lc = load_key
            bm_dict = LINE_dict.get((member, story),{})
            el_ID = bm_dict.get('ID')
            # el_Ds and int_node_list are present for split beams
            el_IDs = bm_dict.get('ELEMENT_IDS',[]) # if multiple elements
            int_node_list = bm_dict.get('INTERMEDIATE_NODES',[])
            # Get load case ID
            lc_ID = LOADCASE_dict.get(lc,{}).get('ID')
            #LC_ID_lookup_dict[lc] # reverse lookup not required
            
            # 
            if (el_ID is not None) and (len(el_IDs) == 0):
                for load_dict in load_list:
                    f_dir = {'GRAV': ('GLOBAL', 'Z', -1)}.get(load_dict.get('DIR'), None)
                    if f_dir and load_dict.get('TYPE') == 'UNIFF':
                        load_value = load_dict.get('DATA')[0][1]
                        ostr = [str(val) for val in ['LOAD_BEAM_UDL.2', '', el_ID, lc_ID, 
                                f_dir[0], 'NO', f_dir[1], f_dir[2] * load_value]]
                        gwa.write('\t'.join(ostr) + '\n') 
                        #pass
                    elif f_dir and load_dict.get('TYPE') == 'TRAPF':
                        (pos_1, value_1), (pos_2, value_2) = load_dict.get('DATA')
                        #value_2
                        ostr = [str(val) for val in ['LOAD_BEAM_PATCH.2', '', el_ID, lc_ID, f_dir[0], 
                                'NO', f_dir[1], -1 * pos_1, f_dir[2] * value_1, -1 * pos_2, f_dir[2] * value_2]]
                        gwa.write('\t'.join(ostr) + '\n') 
            
            # Split loads for split elements
            elif (el_ID is not None) and (len(el_IDs) == (len(int_node_list) + 1) > 1):
                #print(f'el_ID: {el_ID} - *OK* - suitable elements found:')
                #print(f'   el_IDs: {el_IDs}, int_node_list: {int_node_list}')
                for load_dict in load_list:
                    f_dir = {'GRAV': ('GLOBAL', 'Z', -1)}.get(load_dict.get('DIR'), None)
                    if (f_dir is not None) and (load_dict.get('TYPE') == 'UNIFF'):
                        load_value = load_dict.get('DATA')[0][1]
                        for el in el_IDs:
                            ostr = [str(val) for val in ['LOAD_BEAM_UDL.2', '', el, lc_ID, 
                                    f_dir[0], 'NO', f_dir[1], f_dir[2] * load_value]]
                            gwa.write('\t'.join(ostr) + '\n') 
                        #pass
                    elif (f_dir is not None) and (load_dict.get('TYPE') == 'TRAPF'):
                        force_form = build_force_line(load_dict.get('DATA'))
                        t_values = [t for t, *_ in int_node_list]
                        forms = interpolate_force_line3(force_form, t_values)
                        for el, form in zip(el_IDs, forms):
                            for ((pos_1, value_1), (pos_2, value_2)) in [(a, b) for a, b in zip(form[::2], form[1::2])]:
                                if value_1 != 0 or value_2 != 0:
                                    ostr = [str(val) for val in ['LOAD_BEAM_PATCH.2', '', el, lc_ID, f_dir[0], 
                                            'NO', f_dir[1], -1 * pos_1, f_dir[2] * value_1, -1 * pos_2, f_dir[2] * value_2]]
                                    gwa.write('\t'.join(ostr) + '\n') 
            else:
                #print(f'el_ID: {el_ID} - no suitable elements found:')
                #print(f'   el_IDs: {el_IDs}, int_node_list: {int_node_list}')
                pass
            #pass

        # ========================
        # =====  Area Loads  =====
        # ========================
        
        #  AREALOAD  "F42"  "MEP02"  TYPE "UNIFF"  DIR "GRAV"  LC "SW"  FVAL 0.001
        #  AREALOAD  "F7"  "L3"  TYPE "UNIFLOADSET"  "Lobby"
        #  SHELLUNIFORMLOADSET "Lobby"  LOADPAT "LLNR"  VALUE 0.004
        #  SHELLUNIFORMLOADSET "Lobby"  LOADPAT "SDL"  VALUE 0.0034
 

        # LOAD_2D_FACE.2 | name | list | case | axis | type | proj | dir | value(n) | r | s
        # LOAD_2D_FACE.2 <insert example>
        # type: CONS (constant) | GEN (at each corner) | POINT (one value)
        # proj: YES | NO
        # dir: X | Y | Z
        # value: load values (N/m2 !?)
        # r, s: load position for POINT

        LC_ID_lookup_dict = {v['ID']: k for k, v in LOADCASE_dict.items()}
        print(LC_ID_lookup_dict)
        print(LOADCASE_dict)

        print('len of AREA_LOAD_dict', len(AREA_LOAD_dict))
        for load_key, load_list in AREA_LOAD_dict.items():
            member, story, lc = load_key
            sh_ID = AREA_dict.get((member, story),{}).get('ID')
            lc_ID = LOADCASE_dict.get(lc,{}).get('ID')
            #LC_ID_lookup_dict[lc] # reverse lookup not required
            if sh_ID:
                for load_dict in load_list:
                    f_dir = {'GRAV': ('GLOBAL', 'Z', -1)}.get(load_dict.get('DIR'), None)
                    if f_dir and load_dict.get('TYPE') == 'UNIFF':  # 'CONS' - uniform load
                        load_type = 'CONS'
                        load_value = load_dict.get('DATA')[0][1]
                        # LOAD_2D_FACE.2 | name | list | case | axis | type | proj | dir | value(n) | r | s
                        ostr = [str(val) for val in ['LOAD_2D_FACE.2', '', bm_max + sh_ID, lc_ID, 
                                f_dir[0], load_type, 'NO', f_dir[1], f_dir[2] * load_value]]
                        gwa.write('\t'.join(ostr) + '\n') 
                        #pass
                    elif f_dir and load_dict.get('TYPE') == 'point_designation_here':  # 'POINT' 
                        load_type = 'POINT'
                        load_value = load_dict.get('DATA')[0][1]
                        # LOAD_2D_FACE.2 | name | list | case | axis | type | proj | dir | value(n) | r | s
                        ostr = [str(val) for val in ['LOAD_2D_FACE.2', '', bm_max + sh_ID, lc_ID, 
                                f_dir[0], load_type, 'NO', f_dir[1], f_dir[2] * load_value]]
                        gwa.write('\t'.join(ostr) + '\n') 
                    #elif f_dir and load_dict.get('TYPE') == 'node_designation_here':  # 'GEN' 
                    #    load_type = 'GEN'
                    #    value_1, value_2, value_3, value_4 = load_dict.get('DATA')
                    #    # LOAD_2D_FACE.2 | name | list | case | axis | type | proj | dir | value(n) | r | s
                    #    ostr = [str(val) for val in ['LOAD_2D_FACE.2', '', bm_max + sh_ID, lc_ID, f_dir[0], 
                    #            load_type, 'NO', f_dir[1], -1 * pos_1, f_dir[2] * value_1, -1 * pos_2, f_dir[2] * value_2]]
                    #    gwa.write('\t'.join(ostr) + '\n') 

        # ==============================
        # ===== Load Combinations  =====
        # ==============================
        
        # ** Writing Load Combinations to GWA **
        # COMBINATION | case | name | desc | bridge | note
        # LOAD_COMBO_dict

        gwa.write('END\n')


# =======================================
# =============== Standalone ==============

def write_GWA_model(
    GWApath = 'delme.gwa', 
    node_list = [],
    line_list = [],
    NODE_dict = {}, 
    LINE_dict = {}, 
    FRAME_PROPS_dict={},
    SPRING_PROPS_dict = {},
    AREA_dict = {}, 
    SHELL_PROPS_dict={},
    units = Units('N', 'm', 'C')
    ):
    """
    Quick write:
        node_list = [(1, 2.3, 3.2, 1.1), (2, 3.4, 7.1, 1.1), ...]
        line_list = [(2, 1, 2), (4, 2, 3), ...]
        OR
        node_list = [('N1', 2.3, 3.2, 1.1), ('N2', 3.4, 7.1, 1.1), ...]
        line_list = [('B2', 'N1', 'N2'), ('B4', 'N2', 'N3'), ...]
    NODE_dict key = node name / number, contains keys: 'ID', 'COORDS'
    LINE_dict key = beam node / number, contains keys: 'ID', 'N1, 'N2', ['prop_ID']
    """
    if node_list and not NODE_dict:
        for i, (id, x, y, z) in enumerate(node_list):
            if is_numeric(id):
                NODE_dict[id] = {'ID': id, 'COORDS': (x,y,z)}
            else:
                NODE_dict[id] = {'ID': i+1, 'COORDS': (x,y,z)}
    
    if line_list and not LINE_dict:
        for i, (id, n1, n2, *args) in enumerate(line_list):
            if len(args) > 1:
                p_ID = args[0]
            else:
                p_ID = 1
            if is_numeric(id):
                LINE_dict[id] = {'ID': id, 'N1': n1, 'N2': n2, 'prop_ID': p_ID}
            else:
                LINE_dict[id] = {
                    'ID': i+1, 
                    'N1': NODE_dict.get(n1,{}).get('ID',1), 
                    'N2': NODE_dict.get(n2,{}).get('ID',1)}
                print(LINE_dict[id])

    with open(GWApath, 'w') as gwa:
        # ** Writing initial lines to GWA **
        # NB "\xb0", "\xb2", "\xb3" are degree sign, squared and cubed symbols
        gwa.write(r'! This file was originally written by a Python script by the E2KtoJSON_code library' + '\n')

        if NODE_dict:
            write_node(gwa, NODE_dict, SPRING_PROPS_dict)

        if LINE_dict:
            write_beam(gwa, LINE_dict, FRAME_PROPS_dict, units=units)
        
        if AREA_dict:
            write_shells(gwa, AREA_dict, SHELL_PROPS_dict, bm_max=len(LINE_dict))

        gwa.write('END\n')
    if exists(GWApath):
        print(f'{GWApath} written')


def write_node(gwa, NODE_dict, SPRING_PROPS_dict={}):
    """
    'ID', 'NAME', 'COORDS', 'RESTRAINT', 
    """
    # =======================
    # =====    Nodes    =====
    # =======================
    # POINTASSIGN  "3895"  "BASE"  RESTRAINT "UX UY"  SPRINGPROP "PSpr1"  DIAPH "S1FLEX"
    # NODE.2 | num | name | colour | x | y | z | is_grid { | grid_plane | datum | grid_line_a | grid_line_b } | axis | is_rest { | rx | ry | rz | rxx | ryy | rzz } | is_stiff { | Kx | Ky | Kz | Kxx | Kyy | Kzz }
    # NODE.2    46      NO_RGB  -20.95  3.377499    33.3    NO_GRID 0   REST    1   1   1   0   0   0
    # NODE.3 | num | name | colour | x | y | z | restraint | axis |
    #                   mesh_size | springProperty | massProperty | damperProperty
    # NODE.3	2	name	NO_RGB	8.5	2.3	0.1	xyzxxyy	GLOBAL	0	1
    for i, (n_name, n_dict) in enumerate(NODE_dict.items()):
        nid = n_dict.get('ID', try_numeric(n_name) if is_numeric(n_name) else i)
        restr = n_dict.get('RESTRAINT', None)
        fixity = [set_restraints(restr)] if restr else []
        mesh_size = 0
        spr_prop = n_dict.get('SPRINGPROP')
        spr_ID = [SPRING_PROPS_dict.get(spr_prop,{}).get('ID')] if spr_prop else []
        more = []
        if spr_ID:
            more = spr_ID + more
        if more:
            more = [mesh_size] + more
        if more or fixity:
            more = fixity + ['GLOBAL'] + more
        coords = [str(w) for w in n_dict.get('COORDS', ('', '', ''))]
        ostr = ['NODE.3', nid, n_name, 'NO_RGB'] + \
            coords + more
        gwa.write('\t'.join([str(n) for n in ostr]) + '\n')


def write_beam(gwa, LINE_dict, FRAME_PROPS_dict={}, units=Units('N','m', 'C')):
    """
    'ID', 'N1', 'N2', """
    # =======================
    # =====    Beams    =====
    # =======================
    # EL.2 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle 
    #      | is_rls { | rls { | k } } is_offset { | ox | oy | oz } | dummy
    # EL.4 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle 
    #      | is_rls { | rls { | k } } off_x1 | off_x2 | off_y | off_z | dummy | parent
    # EL.4  1       NO_RGB  BEAM    1   1   1   2   0   0   NO_RLS  0   0   0   0
    # NO_RLS | RLS | STIFF  - F, R, K
    
    # ** Writing Beams to GWA **
    bm_max = 1
    for i, (bm_name, bm_dict) in enumerate(LINE_dict.items()):
        bm_ID = bm_dict.get('ID', try_numeric(bm_name) if is_numeric(bm_name) else i)
        N1 = bm_dict.get('N1', '')
        N2 = bm_dict.get('N2', '')
        bm_angle = bm_dict.get('ANG', 0)
        frame_prop_name = bm_dict.get('SECTION', None)
        fp_dict = FRAME_PROPS_dict.get(frame_prop_name, {})
        prop_num = fp_dict.get('ID', bm_dict.get('prop_ID', 1))
        
        # Offsets
        # rigid_zone = bm_dict.get('RIGIDZONE', 0) # not implemented
        offset_sys = bm_dict.get('OFFSETSYS', None)
        
        offsets = [bm_dict.get(offset, 0) for offset in 
                    ('LENGTHOFFI', 'OFFSETYI', 'OFFSETYI', 
                    'LENGTHOFFJ', 'OFFSETYJ', 'OFFSETYJ')]
        
        cardinal_point = bm_dict.get('CARDINALPT', 0)
        if cardinal_point:
            length_unit = fp_dict.get('UNITS')
            u_conv = units_conversion_factor((length_unit, units.length)) if length_unit else 1.0
            D, B, CY, CZ = [u_conv * fp_dict.get(dim, 0) for dim in ('D','B', 'C3', 'C2')]
            off_y, off_z = cardinal_points_offsets(cardinal_point, D, B, CY=0, CZ=0)
        else:
            off_y, off_z = 0, 0

        if any(offsets):
                # NB OFFSETYI etc and offset_sys are not implemented
            offsets = set_offsets(offsets, offset_sys, off_y, off_z)
        else:
            offsets = (0, 0, off_y, off_z)
        offsets_txt = '\t'.join([str(n) for n in offsets])
        
        # Releases
        release = bm_dict.get('RELEASE', '')   # "TI M2I M3I"
        
        if release:
            releases_txt = 'RLS\t' + set_releases(release)
        else:
            releases_txt = 'NO_RLS'
        
        # NB GSA cannot do property modifications for individual members
    
        ostr4 = [str(val) for val in ['EL.4', str(bm_ID), bm_name, 'NO_RGB', 'BEAM', prop_num, prop_num, 
                N1, N2, '', bm_angle]]
        gwa.write('\t'.join(ostr4 + [releases_txt] + [offsets_txt]) + '\n')
        bm_max = max(bm_max, bm_dict.get('ID', 0))


def write_shells(gwa, AREA_dict, SHELL_PROPS_dict={}, bm_max=1):
    """
    'ID', 'Name', 'NumPts', 'N1', 'N2'... 
    """
    # ========================
    # =====    Shells    =====
    # ========================
    # EL.2 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle | is_rls { | rls { | k } } is_offset { | ox | oy | oz } | dummy
    # EL.4  1       NO_RGB  BEAM    1   1   1   2   0   0   NO_RLS  0   0   0   0
    # ** Writing Beams to GWA **
    for sh_name, sh_dict in AREA_dict.items():
        sh_ID = sh_dict.get('ID', '')
        num_pts = sh_dict.get('NumPts')
        nodes = [sh_dict.get('N'+str(n+1), '') for n in range(num_pts)]
        nodes_string = '\t'.join([str(n) for n in nodes])
        #sh_angle = sh_dict.get('ANG', 0)
        
        # These should be added to assemblies
        #pier = sh_dict.get('PIER', None)
        #spandrel = sh_dict.get('SPANDREL', None)
        #diaphragm = sh_dict.get('DIAPH', None)
        
        shell_prop_name = sh_dict.get('SECTION', None)
        sp_dict = SHELL_PROPS_dict.get(shell_prop_name, {})
        prop_num = sp_dict.get('ID', 1)
        if num_pts == 3:
            #gwa.write('EL.2\t{:d}\t{:s}\t\t{:s}\t{:d}\t{:d}\t{}\n'.format(
            #    bmax + a + 1, area_name, 'TRI3', prop_num, prop_num, pts_txt))
            ostr = ['EL.2', bm_max + sh_ID, sh_name, 'NO_RGB', 'TRI3', 
                prop_num, prop_num, nodes_string]
            gwa.write('\t'.join([str(val) for val in ostr]) + '\n')
        elif num_pts == 4:
            ostr = ['EL.2', bm_max + sh_ID, sh_name, 'NO_RGB', 'QUAD4', 
                prop_num, prop_num, nodes_string]
            gwa.write('\t'.join([str(val) for val in ostr]) + '\n')
        else:
            # TODO: may want to create members with polylines > quads
            #el_type = None
            pass
