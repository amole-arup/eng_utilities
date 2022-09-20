"""Utilities for generating GWA text files from the processed
dictionary generated from ETABS text files (E2K, $ET)

Notes:
1. Shell property modification factors are different in GSA
2. 

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
    Note that the effect of the CARDINALPT is included as cy, cz
    
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
    ('lb', 'yd'): 'lb', ('kip', 'yd'): 'kip',
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


def int_node_filter(int_node_list, tol=1E-4):
    """
    """
    s_list = sorted(int_node_list) # , key=lambda x:x[i])
    #print('s_list', s_list)
    return s_list[:1] + [t2 for t1, t2 in zip(s_list[:-1], s_list[1:]) if
                        (
                            (t1[1:] != t2[1:]) and 
                            ((t2[0]-t1[0]) > tol)
                        )]


def GWA_mat_string(name, mat_dict, num, grav=1, debug=False):
    """Generates GWA text string for steel and concrete materials

    mat_type - STEEL, CONCRETE
    
    """
    if mat_dict.get('W') is not None:
        mat_type = mat_dict.get('DESIGNTYPE', 'Unknown')
        wt_density = mat_dict.get('W', 0)
    else:
        mat_type = mat_dict.get('TYPE', 'Unknown')
        wt_density = mat_dict.get('WEIGHTPERVOLUME', 0)
    
    if debug: print(f'-- GWA_mat_string function called for {name} (type = {mat_type}):')
    
    rho = wt_density / grav
    cost = mat_dict.get('PRICE', 0)
    alpha = mat_dict.get('A', 0)
    E_des = mat_dict.get('E', 0)
    E_anl = E_des
    nu = mat_dict.get('U', 0)
    G = 0
    damp = 0
    # num = mat_dict.get('ID')
    num_uc, num_sc, num_ut, num_st = 0, 0, 0, 0
    env = 'NO' # or 'YES' + env_param
    
    mat_def = 'MAT_ELAS_ISO'
    Mat_Anal = ['MAT_ANAL.1', mat_type, '', mat_def, 6, E_anl, nu, rho, alpha, G, damp, 0, 0]

    if mat_type.casefold() == 'steel': # Steel
        fy = mat_dict.get('FY', 0)
        fu = mat_dict.get('FU', 0)
        e_y = fy / E_anl
        eps = 0.05 # 0.00 # limit strain

        el_comp, el_tens, plas_comp, plas_tens, fail_comp, fail_tens = e_y, e_y, e_y, e_y, eps, eps
        gamma_f, gamma_e = 1.0, 1.0
        ULS_def = ['MAT_CURVE_PARAM.3', '', 'ELAS_PLAS', el_comp, el_tens, plas_comp, plas_tens, fail_comp, fail_tens, gamma_f, gamma_e]
        SLS_def = ['MAT_CURVE_PARAM.3', '', 'ELAS_PLAS', el_comp, el_tens, plas_comp, plas_tens, fail_comp, fail_tens, gamma_f, gamma_e]
        Mat_Anal = ['MAT_ANAL.1', mat_type, '', mat_def, 6, E_anl, nu, rho, alpha, G, damp, 0, 0]
        Mat = ['MAT.11', name, E_des, fy, nu, G, rho, alpha] + Mat_Anal + [num_uc, num_sc, num_ut, num_st, eps] + ULS_def + SLS_def + [cost, mat_type, env]

        # Steel design props - defaults
        eps_p, Eh = 0, 0
        
        # Final composition of data
        Mat_Steel = ['MAT_STEEL.3', num] + Mat + [fy, fu, eps_p, Eh]

        if debug:
            if min(len(Mat),len(Mat_Anal), len(Mat_Steel)) > 2:
                print(f'--- Steel Design material Generated - {name} (type = {mat_type}, density = {rho:.4g})')
            else:
                print(f'--- Steel Design material NOT Generated - {name} (type = {mat_type}, density = {rho:.4g})')
        
        # Return a tab-separated string
        return '\t'.join([str(a) for a in Mat_Steel])

    elif mat_type.casefold().startswith('conc'):  # Concrete
        # 'FY', 'FC', 'FYS'
        fcu = mat_dict.get('FC', 0)
        fy = mat_dict.get('FY', 0) # not used here
        fys = mat_dict.get('FYS', 0) # not used here
        
        e_fail = 0.05
        eps = 0.05 # 0.00 # limit strain

        el_comp, el_tens, plas_comp, plas_tens, fail_comp, fail_tens = 0, 0, 0, 0, e_fail, e_fail # strain[6]
        gamma_f, gamma_e = 1.5, 1.0
        ULS_def = ['MAT_CURVE_PARAM.3', '', 'RECT_PARABOLA+NO_TENSION', el_comp, el_tens, plas_comp, plas_tens, fail_comp, fail_tens, gamma_f, gamma_e]
        # e_fail = 0.0035
        el_comp, el_tens, plas_comp, plas_tens, fail_comp, fail_tens = 0, 1, 0, 1, 0, 1
        gamma_f, gamma_e = 1.0, 1.0
        SLS_def = ['MAT_CURVE_PARAM.3', '', 'LINEAR+BS8110_PT2', el_comp, el_tens, plas_comp, plas_tens, fail_comp, fail_tens, gamma_f, gamma_e]
        Mat = ['MAT.11', name, E_des, fcu, nu, G, rho, alpha] + Mat_Anal + [num_uc, num_sc, num_ut, num_st, eps] + ULS_def + SLS_def + [cost, mat_type, env]

        # Concrete design props - defaults
        test_type = 'CYLINDER' # 'CUBE' # 
        cement = 'N' # Normal, 'S' Slow, 'R' Rapid Hardening
        fc, fcd, fcdc, fcdt, fcfib, EmEs, n, Emod = fcu, 0, 0, 0, 0, 0, 1, 1
        Conc_strength_props = [test_type, cement, fc, fcd, fcdc, fcdt, fcfib, EmEs, n, Emod]

        eps_peak, eps_max, eps_plat, eps_u, eps_ax, eps_tran, eps_axs = 0, 0, 0, 0.0035, 0.002, 0, 0
        Conc_strain_props = [eps_peak, eps_max, eps_plat, eps_u, eps_ax, eps_tran, eps_axs]

        light, agg, xd_min, xd_max, beta, shrink, confine, fcc, eps_plas_c, eps_u_c = 'NO', 0.02, 0, 1, 1, 0, 0, 0, 0, 0
        Conc_other_props = [light, agg, xd_min, xd_max, beta, shrink, confine, fcc, eps_plas_c, eps_u_c]

        # Final composition of data
        Mat_Conc = ['MAT_CONCRETE.17', num] + Mat + Conc_strength_props + Conc_strain_props + Conc_other_props

        if debug:
            if min(len(Mat),len(Mat_Anal), len(Mat_Conc)) > 2:
                print(f'--- Concrete Design material Generated - {name} (type = {mat_type}, density = {rho:.4g})')
            else:
                print(f'--- Concrete Design material NOT Generated - {name} (type = {mat_type}, density = {rho:.4g})')

        # Return a tab-separated string
        return '\t'.join([str(a) for a in Mat_Conc])




def write_GWA(E2K_dict, GWApath, GSA_ver=10, add_poly=False, debug=False):
    """Generates a basic GWA file (GSA text file format) 
    from the ETABS dict generated by `run_all` in this module
    :param E2K_dict: The dictionary containing ETABS model data that is 
            generated by `run_all` in this module, containing the following dictionaries:
            ['Stories', 'Points', 'LineDict', 'Lines', 'Areas', 'Groups', 'LoadCases', 'LoadCombs']
    :param GWApath: The file name, including path, for the output GWA file
    :param GSA_ver: This should be provided as an integer or float (e.g. 10 or 10.1)
    :param add_poly: Whether to generate new membrane elements for each storey
    :return: GWA file for reading into GSA"""
    
    if debug: 
        print('\n*************************************************')
        print('***** Starting to export E2K_dict to GWA... *****')
        print('*************************************************')
    # ==================================
    # ============= Titles =============
    # ==================================
    # TITLE | title | sub-title | calc | job_no | initials
    if debug: print('\n===== Generating Title Information =====')
    title1_keys = E2K_dict.get('CONTROLS', {}).get('TITLE1',{}).keys() #'Title 1'
    title1 = list(title1_keys)[0] if title1_keys else ''
    title2_keys = E2K_dict.get('CONTROLS', {}).get('TITLE2',{}).keys() #'Title 2'
    title2 = list(title2_keys)[0] if title2_keys else ''
    
    GSA_num = check_GSA_ver(GSA_ver)
    # =================================
    # ============= Units =============
    # =================================
    
    units = E2K_dict['UNITS']
    length_tolerance = 0.001 *units_conversion_factor(('m', units.length))
    angle_tolerance = 1E-6
    if debug: 
        print('\n===== Defining Units Information =====')
        print(f'Units are {units}')
        print(f'Length tolerance is {length_tolerance} {units.length}')
        print(f'Angle tolerance is {angle_tolerance} radians')
        
    force_factor = units_conversion_factor(('N', units.force))
    length_factor = units_conversion_factor(('m', units.length))
    stress_factor = force_factor / length_factor**2
    stress_units = set_GSA_pressure_units(units.force, units.length)
    mass_factor = force_factor / length_factor
    mass_units = set_GSA_mass_units(units.force, units.length)
    grav_dict = {'m': 9.80665, 'cm': 980.665, 'mm': 9806.65, 'in': 32.2, 'ft': 386.4, 'yd': 1159.2}
    # if US units, then mass units = force units and grav = 1.0
    grav = 1.0 if (units.force == mass_units) else grav_dict.get(units.length, 9.80665)
    
    
    #print('F units:', units.force, force_factor, ', L units:', units.length, length_factor)
    #print('S units:', stress_units, stress_factor, ', M units:', mass_units, mass_factor)
    #print('T units:', "\xb0" + units.temperature, 1.0, '\n')
    if debug and units.force not in ('N', 'kN', 'MN', 'lb', 'ton'):
        print(f'**Non-standard units**: \nCheck derived unit and factors based on {units}:')
        print(f'\tStress units: {stress_units}, {stress_factor}\n\tMass units: {mass_units}, {mass_factor}')
    

    # ===========================================
    # ======== Setting up dictionary refs =======
    # ===========================================
    
    if debug: print('\n===== Setting up references to dictionaries in E2K_dict =====')    
    # SLAB_PROPS_dict, DECK_PROPS_dict, WALL_PROPS_dict, ('SLAB PROPERTIES', 'SHELLPROP'), ('DECK PROPERTIES', 'SHELLPROP'), ('WALL PROPERTIES', 'SHELLPROP'),
    MAT_PROPS_dict, FRAME_PROPS_dict, SHELL_PROPS_dict, SPRING_PROPS_dict = [
        E2K_dict.get(k1,{}).get(k2,{}) for k1, k2 in (
        ('MATERIAL PROPERTIES', 'MATERIAL'), 
        ('FRAME SECTIONS', 'FRAMESECTION'), 
        ('SHELL PROPERTIES', 'SHELLPROP'),
        ('POINT SPRING PROPERTIES', 'POINTSPRING'),
    )]

    STORY_dict, Story_List_dict, DIAPHRAGMS_dict, DIAPHRAGM_GROUPS_dict, DIAPHRAGM_LOOPS_dict = [
        E2K_dict.get(k1,{}).get(k2,{}) for k1, k2 in (
        ('STORIES - IN SEQUENCE FROM TOP', 'STORY'),
        ('STORIES - IN SEQUENCE FROM TOP', 'Story_Lists'),
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

    
    # ============================================
    # ======== Opening GWA File for Writing =======
    # ============================================
    
    if debug: 
        print(f'\n===== Opening GWA file for writing =====')    
        print(f'== {GWApath} ==')    
        print(f'========================================')    
    
    with open(GWApath, 'w', encoding='utf8') as gwa:
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
        # ** Creating Materials Database and writing analysis materials to GWA ** 
        if debug: print(f'\n===== Writing {len(MAT_PROPS_dict)} Mat Properties to GWA =====')    
        # if ETABS_dict.get('MatProps'):
        
        ETABS2GWA_mat_type_dict = {
            'Steel': 'STEEL', 'steel': 'STEEL', 'STEEL': 'STEEL',
            'Coldformed': 'STEEL', 'coldformed': 'STEEL', 'COLDFORMED': 'STEEL',
            'Rebar': 'REBAR', 'rebar': 'REBAR', 'REBAR': 'REBAR',
            'Tendon': 'REBAR', 'tendon': 'REBAR', 'TENDON': 'REBAR',
            'Concrete': 'CONCRETE', 'concrete': 'CONCRETE', 'CONCRETE': 'CONCRETE', 
            'Aluminum': 'ALUMINIUM', 'aluminum': 'ALUMINIUM', 'ALUMINUM': 'ALUMINIUM', 
            }
        
        # MAT_STEEL.3 | num | <mat> | fy | fu | eps_p | Eh
        # MAT_ANAL | num | MAT_ELAS_ISO | name | colour | 6 | E | nu | rho | alpha | G | damp |
        for m_name, m_dict in MAT_PROPS_dict.items():
            m_ID = m_dict.get('ID', '')
            E_val = m_dict.get('E', 0)
            nu = m_dict.get('U', 0)
            rho_w = m_dict.get('W', 0) if m_dict.get('W') is not None else m_dict.get('WEIGHTPERVOLUME', 0)
            rho_m = rho_w / grav
            m_dict['GSA_weight_density'] = rho_w
            m_dict['GSA_mass_density'] = rho_m
            alpha = m_dict.get('A', 0)
            mat_type = m_dict.get('DESIGNTYPE') if m_dict.get('W') is not None else m_dict.get('TYPE', 0)
            # MAT_ANAL | num | MAT_ELAS_ISO | name | colour | 6 | E | nu | rho | alpha | G | damp |
            ostr = [str(val) for val in ['MAT_ANAL', m_ID, 'MAT_ELAS_ISO', m_name, 
                    'NO_RGB', '6', E_val, nu, rho_m, alpha]]
            gwa.write('\t'.join(ostr) + '\n')
        
        # Generate list of design materials & write to GWA file
        GWA_mat_string_list = []
        steel_ID = 0
        conc_ID = 0

        design_mat_lookup = {}
        for name, mat_dict in MAT_PROPS_dict.items():
            m_ID = mat_dict.get('ID', '')
            if mat_dict.get('W') is not None: # To detect ETABS data format
                mat_type = mat_dict.get('DESIGNTYPE', '')
            else:
                mat_type = mat_dict.get('TYPE', '')
            mat_dict['GWA_Type'] = ETABS2GWA_mat_type_dict.get(mat_type, 'OTHER')
            
            if debug: print(f'Material = {name}; Material_Type = {mat_type}')
            if mat_type.casefold().startswith('conc'):
                conc_ID += 1
                design_mat_lookup[m_ID] = str(conc_ID)
                GWA_mat_string_list.append(GWA_mat_string(name, mat_dict, num=conc_ID, grav=grav, debug=debug))
            elif mat_type.casefold() == 'steel':    
                steel_ID += 1
                design_mat_lookup[m_ID] = str(steel_ID)
                GWA_mat_string_list.append(GWA_mat_string(name, mat_dict, num=steel_ID, grav=grav, debug=debug))
            else:
                design_mat_lookup[m_ID] = ''

        # write to file
        for GWA_line in GWA_mat_string_list:
            gwa.write(GWA_line + '\n')
        

        # ============================
        # ===== Frame Properties =====
        # ============================
        if debug: print(f'\n===== Writing {len(FRAME_PROPS_dict)} Frame Properties to GWA =====')    
        # NB The information below is quite confusing... use PROP_SEC & SECTION_MOD
        # or whatever works... TODO: Try inserting into GWA file that works...
        # 
        # ** SEC, SEC_PROP & SEC_MOD were the basic formats for 8.7
        # SEC | num | name | mat | desc | area | Iyy | Izz | J | Ky | Kz 
        # SEC_PROP | num | prop | name | mat | desc | area | Iyy | Izz | J | Ky | Kz 
        # [prop - property type = BASE | ANAL]
        # SEC_MOD | num | area_f | area_i | I11_f | I11_i | I22_f | I22_i | J_f | J_i |K1_f | K1_i | K2_f | K2_i | mass | stress | principal 
        # [ _f is 0 if _i is a replacement, otherwise it is a scaling factor]
        # [mass adjustment with area = YES | NO, use principal axes YES | NO]
        # [stress calculation option = NO_MOD | USE_UNMOD | USE_MOD]
        # 
        # ** Defined in GSA 9.0 and later
        # PROP_SEC	13	My_Explicit_Section	NO_RGB	2	EXP	NO	NA	0.000000	PROP
        # 	0.0248	0.0019	0.0019	0.0038	0.5	0.5	NO_MOD_PROP	NO_PLATE	NO_J
        # PROP_SEC.1 | num | name | colour | anal | desc | prin | type | cost 
        # | is_prop { | area | I11 | I22 | J | K11 | K22 } - only required if desc is EXP
        # | is_mod { | area_to_by | area_m | I11_to_by | I11_m | I22_to_by | I22_m | J_to_by | J_m | K11_to_by | K11_m | K22_to_by | K22_m 
        # | mass | stress } | is_env { | energy | carbon | recycle | user }
        # PROP_SEC.2 | num | name | colour | type | desc | anal | mat | grade |
        # PROP_SEC.3 | num | name | colour | mat | grade | anal | desc | cost | ref_point | off_y | off_z |
        # ref_point is base reference point: []
        # GSA
        # ref_points = ['CENTROID', 'TOP_LEFT', 'TOP_CENTRE', 'TOP_RIGHT', 'MID_LEFT', 'MID_RIGHT', 'BOT_LEFT', 'BOT_CENTRE', 'BOT_RIGHT']
        # ref_points_dict = {10:'CENTROID', 7:'TOP_LEFT', 8:'TOP_CENTRE', 9:'TOP_RIGHT', 4:'MID_LEFT', 5:'CENTROID', 
        #            6:'MID_RIGHT', 1:'BOT_LEFT', 2:'BOT_CENTRE', 3:'BOT_RIGHT', 11:'CENTROID'} # mapping
        #
        # ETABS Cardinal Points
        #cardinal_pts = {1:'Bottom left', 2:'Bottom center', 3:'Bottom right', 4:'Middle left', 5:'Middle centre', 
        #            6:'Middle right', 7: 'Top left', 8:'Top center', 9:'Top right', 10:'Centroid', 11:'Shear center'}
        # [grade is design material, anal is analysis material]
        # [ref_point - base reference point = CENTROID,TOP_LEFT, TOP_CENTRE, TOP_RIGHT, MID_LEFT, MID_RIGHT, BOT_LEFT, BOT_CENTRE, BOT_RIGHT]
        # [type = GENERIC | BEAM | COLUMN, prin not used, type = NA	not-applicable, WELDED welded, ROLLED hot-rolled, FORMED formed]
        # [is_prop = PROP | NO_PROP, is_mod = MOD_PROP | NO_MOD_PROP (TO or BY), mass = YES | NO, stress = NO_MOD | USE_UNMOD | USE_MOD, is_env = ENV | NO_ENV]
        # SECTION_MOD | ref | name | mod | centroid | stress | opArea | area | prin 
        #   | opIyy | Iyy | opIzz | Izz | opJ | J | opKy | ky | opKz | kz | opVol | vol | mass
        # [Use with the PROP_SEC definitions]
        # [op = BY | TO,  prin = UV | YZ,  mod = GEOM | STIFF,  centroid = SEC | CEN,  stress = NONE | UNMOD | MOD]
        #
        # Newer definitions:
        # SECTION.2 | ref | name | memb:EXPLICIT | mat 
        # | matType | mat | area | Iyy | Izz | J | ky | kz | vol | mass | fraction | cost
        # SECTION_COMP | ref | name | matAnal | matType | matRef | desc | offset_y | offset_z | rotn | reflect | pool
        # SECTION.7 | ref | colour | name | memb | pool | point
        #  | refY | refZ | mass | fraction | cost | left | right | slab | num
        # [memb GENERIC | 1D_GENERIC??]
        # [point = CENTROID | TOP_LEFT | TOP_CENTRE | TOP_RIGHT | MIDDLE_LEFT | MIDDLE_RIGHT | BOTTOM_LEFT | BOTTOM_CENTRE | BOTTOM_RIGHT]
        # SECTION.7	96	NO_RGB	COL_3B(1)_1300DIA_SRC	1D_GENERIC	0	CENTROID
        # 	0	0	0	1	0	0	0	0	1	
        # SECTION.7	134	NO_RGB	Explicit_Section	1D_GENERIC	0	CENTROID
        # 	0	0	0	1	0	0	0	0	1	
        #   SECTION_COMP.4		6	STEEL	0	
        #   EXP 3620 24351337.053407 6200995.3959484 104266.50680667 0.24226277894978 0.58715758620862	0	0	0	NONE	0	NONE	0	
        #   SECTION_STEEL.2	0	1	1	1	0.4	NO_LOCK	UNDEF	UNDEF	0	0	NO_ENVIRON
        # SECTION_ANAL.4	96	COL_3B(1)_1300DIA_SRC	GEOM	SEC	NONE
        #   	BY	1.23	YZ	BY	1.24	BY	1.25	BY	1	BY	1	BY	1	BY	1.23	0
        # ** Creating Properties Database and writing to GWA **

        mat_types = ['GENERIC', 'STEEL', 'CONCRETE', 'ALUMINIUM', 'GLASS', 'FRP', 'TIMBER']
        for fp_name, fp_dict in FRAME_PROPS_dict.items():
            mat_name = fp_dict.get('MATERIAL') if (fp_dict.get('ENCASEMENTMATERIAL') is None) else fp_dict.get('ENCASEMENTMATERIAL')
            m_dict = MAT_PROPS_dict.get(mat_name, {})
            mat_ID = m_dict.get('ID', 1)
            fp_ID = fp_dict.get('ID', '')
            cost = fp_dict.get('PRICE', 0)
            desc = fp_dict.get('GWA', 'EXPLICIT')
            mat_type = m_dict.get('TYPE','').upper()
            mat_type = mat_type if (mat_type in mat_types) else 'GENERIC'
            mat_dens = m_dict.get('GSA_mass_density', None)
            if mat_dens is None:
                lineal_density = 0
                if debug: print(f'frame_prop: {fp_name}: {mat_name} has no density specified')
            else:
                lineal_density = m_dict['GSA_mass_density'] * fp_dict.get('A', 0)
            # PROP_SEC.2 | num | name | colour | type | desc | anal | mat | grade |
            ostr2 = [str(val) for val in ['PROP_SEC.2', fp_ID, fp_name, 'NO_RGB', '', 
                     desc, mat_ID, mat_type]]
            # PROP_SEC.3 | num | name | colour | mat | grade | anal | desc | cost | ref_point | off_y | off_z |
            ostr3 = [str(val) for val in ['PROP_SEC.3', fp_ID, fp_name, 'NO_RGB', mat_type, 
                     design_mat_lookup.get(mat_ID,''), mat_ID, desc, cost, 'CENTROID', 0, 0]]
            gwa.write('\t'.join(ostr3) + '\n')

            # Modifiers if required
            # Area, Iyy, Izz, J, Ky, Kz, Vol
            modifiers = ['AMOD', 'I3MOD', 'I2MOD', 'JMOD', 'AS3MOD', 'AS2MOD', 'WMOD'] #, 'MMOD']
            if any(fp_dict.get(val, None) for val in modifiers):
                mod = 'STIFF' # 'GEOMETRY' # 
                centroid = 'CEN' # 'SEC'
                stress = 'MOD' # 'UNMOD', 'NONE'
                prin = 'YZ' # 'UV'
                # GSA modifier is either 'BY' or 'TO'
                
                mod_txt = [f'BY\t{fp_dict.get(val,1):6.4f}' for val in modifiers]
                mass_fac  = fp_dict.get('MMOD', 1.0)
                sec_add_mass = lineal_density * (mass_fac - 1)  # additional non-structural mass/length
                ostr0 = [str(val) for val in ['SECTION_ANAL.4', fp_ID, fp_name, mod, centroid, 
                            stress, mod_txt[0], prin] + mod_txt[1:] + [sec_add_mass]]
                gwa.write('\t'.join(ostr0) + '\n')

        
        # ============================
        # ===== Shell Properties =====
        # ============================
        if debug: print(f'\n===== Writing {len(SHELL_PROPS_dict)} Shell Properties to GWA =====')    
        # for calculating mat_type
        # mat_type_dict = {'steel': 'Steel', 'concrete': 'Concrete'} #, FRP, ALUMINIUM, GLASS, TIMBER, GENERIC, FABRIC}
        shell_type_dict = {'shell': 'SHELL', 'membrane': 'STRESS'}
        # Area properties
        # PROP_2D.1 | num | name | axis | mat | type | thick | mass | bending
        # PROP_2D.8 | num | name | colour | type | axis | mat | mat_type | grade | design | profile | ref_pt | ref_z | mass | flex | shear | inplane | weight |
        # PROP_2D.8 | num | name | colour | LOAD | support | edge
        # PROP_2D.2 | num | name | colour | axis | mat | type | thick | mass | bending | inplane | weight
        # PROP_2D.8	1	S250	NO_RGB	SHELL	LOCAL	1	GENERIC	0	0	250(mm)	CENTROID	0	0	100%	-0%	100%	100%
        for sp_name, sp_dict in SHELL_PROPS_dict.items():
            sp_ID = sp_dict.get('ID', '')
            mat_name = sp_dict.get('MATERIAL', sp_dict.get('CONCMATERIAL', ''))
            m_dict = MAT_PROPS_dict.get(mat_name, {})
            m_ID = m_dict.get('ID', 1)
            shell_colour = 'NO_RGB'
            shell_axis = 'LOCAL'
            desc = sp_dict.get('GWA', 'EXPLICIT')
            shell_type = shell_type_dict.get((sp_dict.get('TYPE', 'shell').lower()))
            thickness = max([sp_dict.get(t, 0) for t in ['SLABTHICKNESS', 'WALLTHICKNESS','DECKTHICKNESS','DECKSLABDEPTH']])
            mat_dens = m_dict.get('GSA_mass_density', None)
            if mat_dens is None:
                mass_per_area = 0
                if debug: print(f'shell_prop: {sp_name}: {mat_name} has no density specified')
            else:
                mass_per_area = m_dict['GSA_mass_density'] * thickness
            
            # PROP_2D.2 | num | name | colour | axis | mat | type | thick | mass | bending | inplane | weight
            #ostr2 = ['PROP_2D.2',str(sp_ID), sp_name, 'NO_RGB', 'LOCAL', str(m_ID),
            #        shell_type, str(thickness), '0.0', '100.0%', '100.0%', '100.0%']
            ostr2 = [str(val) for val in ['PROP_2D.8',sp_ID, sp_name, shell_colour, shell_axis, m_ID,
                    shell_type, thickness, '0.0', '100.0%', '100.0%', '100.0%']]
            
            # PROP_2D.8 | num | name | colour | type | axis | mat | mat_type | grade | design | profile | ref_pt | ref_z | mass | flex | shear | inplane | weight |
            shell_type = 'SHELL'  # 'FABRIC', 'PLATE', 'SHELL', 'CURVED', 'WALL', 'STRESS', 'STRAIN', 'AXI', 'LOAD'
            #shell_axis = 'LOCAL'
            shell_mat = m_ID   #  Number of the analysis material (if mat < 0: layer material, layer thickness, layer angles)
            # ['STEEL', 'CONCRETE', 'FRP', 'ALUMINIUM', 'GLASS', 'TIMBER', 'GENERIC', 'FABRIC']
            mat_type = m_dict.get('GWA_Type', 'GENERIC')  
            grade = design_mat_lookup.get(shell_mat, 0)   # Number of the design material
            design = 0   # design property  ???
            profile = f'{thickness}({units.length})'   #  need to provide units
            ref_pt = 'CENTROID'  # 'TOP_CENTRE', 'BOT_CENTRE' # reference surface
            ref_z = 0  # z offset from reference surface
            # Modifiers...
            # SHELLPROP  "Slab1"  F11MOD 1.001 F22MOD 1.01 F12MOD 1.02 M11MOD 1.03 M22MOD 1.04 M12MOD 1.05 V13MOD 1.06 V23MOD 1.07 MMOD 1.08 WMOD 1.09
            mass_fac = sp_dict.get('MMOD', 1.0)
            shell_add_mass = mass_per_area * (mass_fac - 1)   # additional mass per unit area [kg/m²] this could take account of MMOD
            flex_fac = sum(sp_dict.get(fac, 1.0) for fac in ('M11MOD', 'M22MOD')) / 2.0
            flex_mod = f'{flex_fac * 100}%'   # stiffness modifier for bending
            shear_mod = f"{sp_dict.get('F12MOD', 1.0) * 100}%"   #   stiffness modifier for shear
            inplane_fac = sum(sp_dict.get(fac, 1.0) for fac in ('F11MOD', 'F22MOD')) / 2.0
            inplane_mod = f'{inplane_fac * 100}%'   #  stiffness modifier for inplane stiffness
            weight_fac = sp_dict.get('WMOD', 1.0)
            weight_mod = f'{weight_fac * 100}%'   #  stiffness modifier for weight
            ostr8 = [str(val) for val in ['PROP_2D.8', sp_ID, sp_name, shell_colour, shell_type, shell_axis, 
                    shell_mat, mat_type, grade, design, profile, ref_pt, ref_z, shell_add_mass, flex_mod, shear_mod, 
                    inplane_mod, weight_mod]]
    
            #gwa.write('\t'.join(ostr2) + '\n')
            gwa.write('\t'.join(ostr8) + '\n')
               
        
        # =============================
        # ===== Spring Properties =====
        # =============================
        if debug: print(f'\n===== Writing {len(SPRING_PROPS_dict)} Spring Properties to GWA =====')    
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
        if debug: print(f'\n===== Writing {len(NODE_dict)} Nodes to GWA =====')    
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
        if debug: print(f'\n===== Writing {len(LINE_dict)} Lines to GWA =====')    
        # EL.2 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle 
        #      | is_rls { | rls { | k } } is_offset { | ox | oy | oz } | dummy
        # EL.4 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle 
        #      | is_rls { | rls { | k } } off_x1 | off_x2 | off_y | off_z | dummy | parent
        # EL.4  1       NO_RGB  BEAM    1   1   1   2   0   0   NO_RLS  0   0   0   0
        # NO_RLS | RLS | STIFF  - F, R, K
        # 
        # MEMB.8 | num | name | colour | type (1D) | exposure | prop | group | topology | node | angle | mesh_size | is_intersector | analysis_type 
        #        | fire | limiting_temperature | time[4] | dummy | rls_1 { | k_1 } rls_2 { | k_2 } | restraint_end_1 | restraint_end_2 | AUTOMATIC | load_height | load_ref | is_off { | auto_off_x1 | auto_off_x2 | off_x1 | off_x2 | off_y | off_z }
        # MEMB.8	12	"('B51', 'R/F')"	NO_RGB	1D_GENERIC	ALL	3	3	77 113 72	0	0	0	YES	BEAM	
        #   0	0	0	0	0	0	ACTIVE	FFFFFF	FFFFFF	Free	Free	AUTOMATIC	0	SHR_CENTRE	OFF	MAN	MAN	0	0	0	-0.3
        # member type: BEAM, COLUMN, GENERIC_1D, SLAB, WALL, GENERIC_2D, VOID_CUTTER_1D, VOID_CUTTER_2D	
        bm_type_dict = {'BEAM':'BEAM', 'BRACE':'BEAM', 'COLUMN': 'COLUMN', 'LINE': '1D_GENERIC'}
        # mat material type: STEEL, CONCRETE, FRP, ALUMINIUM, TIMBER, GLASS
        #
        
        
        # ** Writing Beams to GWA **
        bm_ID_max = max(bm_dict.get('ID',0) for bm_dict in LINE_dict.values()) if LINE_dict else 0
        sh_ID_max = max(sh_dict.get('ID',0) for sh_dict in AREA_dict.values()) if AREA_dict else 0
        next_ID = bm_ID_max + sh_ID_max + 1
        
        bm_max = 1
        for bm_name, bm_dict in LINE_dict.items():
            bm_ID = bm_dict.get('ID', '')
            N1 = bm_dict.get('N1', '')
            N2 = bm_dict.get('N2', '')
            bm_angle = bm_dict.get('ANG', 0)
            bm_type = bm_dict.get('MEMTYPE', '')
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
                nds = [N1, N2]
                node_pairs = [(N1, N2)]
                bm_ids = [bm_ID]

            # Offsets
            # rigid_zone = bm_dict.get('RIGIDZONE', 0) # not implemented
            offset_sys = bm_dict.get('OFFSETSYS', None) # needs to be implemented 'LOCAL'
            
            offsets = [bm_dict.get(offset, 0) for offset in 
                       ('LENGTHOFFI', 'OFFSETYI', 'OFFSETZI', 
                        'LENGTHOFFJ', 'OFFSETYJ', 'OFFSETZJ')]
            
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
                # set_releases 
                releases_txt = ['RLS\t' + rel for rel in set_releases(release, segment_num)]
                mem_releases_txt = [rel for rel in set_releases(release, 1)]
            else:
                releases_txt = ['NO_RLS'] * segment_num
                mem_releases_txt = ['FFFFFF', 'FFFFFF']
            
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
            
            # EL.4	224	"('C26', 'R/F')"	NO_RGB	BEAM	2	2	291	78	0	45	NO_RLS	0	0	0	0	DUMMY	37
            for beam_ID, (n_i, n_j), rel_txt in zip(bm_ids, node_pairs, releases_txt):
                dummy = '\tDUMMY' if prop_num == 1 else ''
                ostr4 = [str(val) for val in ['EL.4', str(beam_ID), bm_name, 'NO_RGB', 'BEAM', 
                        prop_num, prop_num, n_i, n_j, '', bm_angle]]
                if n_i != n_j:  # TODO: identify and eliminate the generation of these extra nodes
                    gwa.write('\t'.join(ostr4 + [rel_txt] + [offsets_txt]) + dummy + '\n')
            bm_max = max(bm_max, bm_dict.get('ID', 0))

            # ========================
            # Writing 1D members...
            # MEMB.8 | num | name | colour | type (1D) | exposure | prop | group | topology | node | angle | mesh_size | is_intersector | analysis_type 
            #        | fire | limiting_temperature | time[4] | dummy | rls_1 { | k_1 } rls_2 { | k_2 } | restraint_end_1 | restraint_end_2 | AUTOMATIC | load_height | load_ref | is_off { | auto_off_x1 | auto_off_x2 | off_x1 | off_x2 | off_y | off_z }
            # MEMB.8	12	"('B51', 'R/F')"	NO_RGB	1D_GENERIC	ALL	3	3	77 113 72	0	0	0	YES	BEAM	
            #   0	0	0	0	0	0	ACTIVE	FFFFFF	FFFFFF	Free	Free	AUTOMATIC	0	SHR_CENTRE	OFF	MAN	MAN	0	0	0	-0.3
            el_type = bm_type_dict.get(bm_type,'1D_GENERIC')
            dummy = 'ACTIVE' if dummy == '' else 'DUMMY'
            # MAXSTASPC 500 AUTOMESH "YES"  MESHATINTERSECTIONS "YES"  FLOORMESH "Yes"
            is_intersector = 'YES' if bm_dict.get('AUTOMESH', 0) == 'YES' else 'NO' 
            mesh_size = 0 # bm_dict.get('ID', 0)
            ostr5 = [str(val) for val in ['MEMB.8', str(bm_ID), bm_name, 'NO_RGB', el_type, 
                        'ALL', prop_num, prop_num, ' '.join([str(nd) for nd in nds]), '', 
                        bm_angle, mesh_size, is_intersector, 'BEAM', 0, 0, 0, 0, 0, 0, dummy, ]]
            # TODO: these additional pieces of information do not work - perhaps Free, Free should not be there if NO_RLS
            if any(offsets):
                ostr6 = [str(val) for val in ['Free', 'Free', 'AUTOMATIC', 0, 'SHR_CENTRE', 'OFF', 'MAN', 'MAN'] + \
                    list(offsets) ] 
            else:
                ostr6 = []          
            # TODO: identify and eliminate the generation of extra nodes
            gwa.write('\t'.join(ostr5 + mem_releases_txt + ostr6) + '\n')

        if debug: print(f'     Max beam ID is: {bm_max}')
        
        
        # ========================
        # =====    Shells    =====
        # ========================
        if debug: print(f'\n===== Writing {len(AREA_dict)} Areas to GWA =====')    
        # parent is the parent member number
        # EL.2 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle | is_rls { | rls { | k } } is_offset { | ox | oy | oz } | dummy
        # EL.4 | num | name | colour | type | prop | group | topo() | orient_node | orient_angle | is_rls { | rls { | k } } off_x1 | off_x2 | off_y | off_z | dummy | parent
        # EL.4  1    el_name   NO_RGB  BEAM    1   1   1   2   0   0   NO_RLS  0   0   0   0
        # element type: BAR, BEAM, TIE, STRUT, SPRING, GRD_SPRING, LINK, DAMPER, GRD_DAMPER, CABLE, SPACER, MASS, GROUND, TRI3, TRI6, QUAD4, QUAD8, BRICK8
        # 
        # member type: BEAM, COLUMN, GENERIC_1D, SLAB, WALL, GENERIC_2D, VOID_CUTTER_1D, VOID_CUTTER_2D	
        sh_type_dict = {'FLOOR':'SLAB', 'RAMP':'SLAB', 'PANEL': 'WALL', 'AREA': '2D_GENERIC'}
        # mat material type: STEEL, CONCRETE, FRP, ALUMINIUM, TIMBER, GLASS
        #
        # ** Writing Beams to GWA **
        memb_i = 1
        sh_max = bm_max
        for sh_name, sh_dict in AREA_dict.items():
            sh_ID = sh_dict.get('ID', '')
            sh_type = sh_dict.get('MEMTYPE', '')
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
            
            # ======================================================
            # Edited to cause all shells to be written to 2D members
            # MEMB.8 | num | name | colour | type (2D) | exposure | prop | group | topology | node | angle | mesh_size | is_intersector | analysis_type 
            #        | fire | limiting_temperature | time[4] | dummy | off_z | off_auto_internal | reinforcement2d |
            # MEMB.8	1	Shell F2 (ID: 48) @ G/F	NO_RGB	SLAB	ALL	1	1	214 216 28 29 30 215	0	0	0	NO	LINEAR
            # 	0	0	0	0	0	0	ACTIVE	0	NO	REBAR_2D.1	0.03	0.03	0
            # OBJMESHTYPE ("AUTOMESH"), MESHAT ("BEAMS" | "WALLS"), MAXMESHSIZE
            mesh_size = sh_dict.get('MAXMESHSIZE', 0)
            is_intersector = 'YES' if sh_dict.get('OBJMESHTYPE', '') == 'AUTOMESH' else 'NO'
            analysis_type = 'LINEAR'
            # dummy = '\tDUMMY' if prop_num == 1 else ''
            dummy = 'DUMMY' if prop_num == 1 else 'ACTIVE'
            
            if num_pts > 2:
                # Create polygonal members for n-gon with n > 4 
                el_type = sh_type_dict.get(sh_type,'2D_GENERIC')
                if sh_dict.get('OPENING', '') == 'YES':
                    el_type = 'VOID_CUTTER_2D'
                el_name, story_name = sh_name
                pname = f'Shell {el_name} (ID: {sh_ID}) @ {story_name}'
                # MEMB.8	1	name	NO_RGB	SLAB	ALL	1	1	node_list	0	0	0	YES
                ostr = [str(val) for val in ['MEMB.8', bm_max + sh_ID , pname, 'NO_RGB', el_type, 'ALL', prop_num, 1, 
                        nodes_string.replace('\t', ' '), 0, 0, mesh_size, is_intersector, analysis_type, 
                        0,0,0,0,0,0, dummy]]
                gwa.write('\t'.join(ostr) + '\n')
                #memb_i += 1
            sh_max = bm_max + sh_ID
        
        memb_max = sh_max
        if debug: print(f'     Max shell_ID is: {sh_max}')
        
        
        # =======================
        # =====   Storeys   =====
        # =======================
        if debug: print(f'\n===== Writing {len(STORY_dict)} Story Definitions to GWA =====')    
        
        # ** Writing Storey Grid Planes to GWA **
        # GSA 10   GRID_PLANE.4  1   SPIRE-11    STOREY  0   247.967 0   0
        # GSA 8.7  GRID_PLANE.1   1   Ground floor    0   ELEV    0.000000    all     ONE 0.000000    0   0.0100000   STOREY  PLANE_CORNER    0.000000    0.000000
        # GSA 8.7  GRID_PLANE.1\t{num}\t{name}\t{axis}\tELEV\t{elev}\tall\tONE\t0.0\t0\t.01\tSTOREY\tPLANE_CORNER\t0.0\t0.0\n
        
        for s, (story, data) in enumerate(STORY_dict.items()):
            abs_elev = data.get('ABS_ELEV')
            if GSA_num < 10:
                axis = 0
                ostr = ['GRID_PLANE', str(s+1), str(story), str(axis), 'ELEV', str(abs_elev), 
                        'all\tONE\t0.0\t0\t.01\tSTOREY\tPLANE_CORNER\t0.0\t0.0']
                gwa.write('\t'.join(ostr) + '\n')
            else:
                ostr = ['GRID_PLANE.4', str(s+1), str(story), 'STOREY\t0',str(abs_elev),'0\t0']
                gwa.write('\t'.join(ostr) + '\n')
            
        
        # ======================
        # =====   Groups   =====
        # ======================
        if debug: print(f'\n===== Writing {len(GROUPS_dict)} Groups to GWA =====')    
        
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
                    #gwa.write('\t'.join(['LIST.1', str(list_id), g_name, 'ELEMENT']))
                    gwa.write('\t'.join(['LIST.1', str(list_id), g_name, 'MEMBER']))
                    gwa.write('\t')
                    gwa.write(GWA_list_shrink(el_list))
                    gwa.write('\n')
                    list_id += 1

        # ========================
        # =====  Diaphragms  =====
        # ========================
        if debug: print(f'\n===== Writing {len(DIAPHRAGM_GROUPS_dict)} Diaphragm Groups to GWA =====')    
        # RIGID.3 | name | primary_node | type | constrained_nodes | stage | parent_member
        # RIGID.3 <insert example>
        #
        # type: ALL, XY_PLANE, PIN, XY_PLANE_PIN, <link>
        
        # ** Writing Diaphragm Constraints to GWA **
        # Add diaphragms based on groups for each floor 
        # DIAPHRAGMS_dict contains listings for each diaphragm
        # e.g. {'D1': {'TYPE': None, 'RIGID': None}, 'D2': None}
        # DIAPHRAGM_GROUPS_dict contains groupings of all points assigned to a diaphragm at a floor
        #   Note that the nodes that are disconnected are also grouped...
        # e.g. {('33F', 'D1'):[(101, '33F'), (104, '33F'), (16, '33F')], ...,
        #            ('3F', 'DISCONNECTED'): [(101, '3F'), (104, '3F'), (27, '3F')], ...}
        
        # list diaphragms sorted on elevation
        d_keys = list(DIAPHRAGM_GROUPS_dict.keys())
        d_keys.sort(key = lambda x : STORY_dict.get(x[0]).get('ABS_ELEV'))
        
        # for diaph_key, diaph_node_list in DIAPHRAGM_GROUPS_dict.items():
        for diaph_key in d_keys:
            diaph_node_list = DIAPHRAGM_GROUPS_dict.get(diaph_key, [])
            n_list = [NODE_dict.get(nd_key,{}).get('ID') for nd_key in diaph_node_list]
            g_name = f'Diaphragm {diaph_key[1]} @ {diaph_key[0]}'
            # LIST | num | name | type | list
            gwa.write('\t'.join(['LIST.1', str(list_id), g_name, 'NODE']))
            gwa.write('\t')
            gwa.write(GWA_list_shrink([n for n in n_list if n is not None]))
            gwa.write('\n')
            list_id += 1

            diaph_dict = DIAPHRAGMS_dict.get(diaph_key[1], {})
            if diaph_key[1] == 'DISCONNECTED':
                diaphragm_type = None
            else:
                if len(diaph_dict.keys()) == 0:
                    diaphragm_type = 'RIGID'
                elif ('RIGID' in diaph_dict.keys()) and (diaph_dict.get('TYPE','') is None):
                    diaphragm_type = 'RIGID'
                else:
                    diaphragm_type = None
                # diaphragm_type = diaph_dict.get('TYPE', 'RIGID')

            g_string = r'"""' + str(g_name) + r'"""'
            if diaphragm_type == 'RIGID':
                d_type = 'XY_PLANE'
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
            if debug: print('Diaphragm polylines: Generating storey extents as a member')
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
                    ostr = [str(val) for val in ['MEMB.8', perim_i , pname, 'NO_RGB', 'SLAB', 'ALL', p_max, 1, loop_string]]
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
        if debug: print(f'\n===== Writing {len(LOADCASE_dict)} Loadcases to GWA =====')    
        
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
            'Reducible Live': 'IMPOSED',
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
        if debug: print(f'\n===== Writing {len(LINE_LOAD_dict)} Line Loads to GWA =====')    
        
        #   LINELOAD  "B2"  "L3"  TYPE "TRAPF"  DIR "GRAV"  LC "SDL"  FSTART 4.55  FEND 4.55  RDSTART 0  RDEND 0.3780311  
        #   LINELOAD  "B2"  "L3"  TYPE "TRAPF"  DIR "GRAV"  LC "SDL"  FSTART 4.55  FEND 4.55  RDSTART 0.3780311  RDEND 1  
        
        # LOAD_BEAM_PATCH.2		1744	2	GLOBAL	NO	Z	0	-4.55	-0.3780311	-4.55
        # LOAD_BEAM_PATCH.2		1744	2	GLOBAL	NO	Z	-0.3780311	-4.55	-1	-4.55
        
        # LOAD_BEAM_UDL.2 | name | list | case | axis | proj | dir | value
        # LOAD_BEAM_UDL.2	Uniform Load	286 287 295 296	2	GLOBAL	NO	Z	-5000
        # pos is negative if relative distance
        # LOAD_BEAM_PATCH.2 | name | list | case | axis | proj | dir | pos_1 | value_1 | pos_2 | value_2


        LC_ID_lookup_dict = {v['ID']: k for k, v in LOADCASE_dict.items()}
        if debug:
            print(LC_ID_lookup_dict)
            print(LOADCASE_dict)

        for load_key, load_list in LINE_LOAD_dict.items():
            member, story, lc = load_key
            bm_dict = LINE_dict.get((member, story),{})
            el_ID = bm_dict.get('ID')
            # el_Ds and int_node_list are present for split beams
            el_IDs = bm_dict.get('ELEMENT_IDS',[]) # if multiple elements
            int_node_list = bm_dict.get('INTERMEDIATE_NODES',[])
            
            # Check if there are problems with the intermediate nodes...
            if debug and len(int_node_list) != len(int_node_filter(int_node_list)):
                print(f'... el: {el_ID}, load key: {load_key}, \n    int_nodes: {int_node_list}')

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
                    elif load_dict.get('TYPE') == 'TEMP':
                        # LOAD_1D_THERMAL.2 | name | list | case | type | value
                        # LOAD_1D_THERMAL.2 | name | item | case | type | pos_1 | value_1 | pos_2 | value_2
                        ltype = 'CONS'
                        load_value = load_dict.get('DATA', 0)
                        ostr = [str(val) for val in ['LOAD_1D_THERMAL', '', el_ID, lc_ID, 
                                ltype , load_value]]
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

                        # Check for problems in the force form
                        if debug and len(force_form) < 1:
                            print(f'...  el: {el_ID}, load key: {load_key} - force form is too short: {force_form}')
                            print(f'    int_nodes: {int_node_list}')
                        
                        try:
                            forms = interpolate_force_line3(force_form, t_values) if (len(force_form) > 1) else force_form
                        except Exception as e:
                            print(f'*** Error: force_form: {force_form}, t_values: {t_values}')
                            print('*** Error message: ', e)
                        
                        for el, form in zip(el_IDs, forms):
                            for ((pos_1, value_1), (pos_2, value_2)) in [(a, b) for a, b in zip(form[::2], form[1::2])]:
                                if value_1 != 0 or value_2 != 0:
                                    ostr = [str(val) for val in ['LOAD_BEAM_PATCH.2', '', el, lc_ID, f_dir[0], 
                                            'NO', f_dir[1], -1 * pos_1, f_dir[2] * value_1, -1 * pos_2, f_dir[2] * value_2]]
                                    gwa.write('\t'.join(ostr) + '\n') 
                    elif load_dict.get('TYPE') == 'TEMP':
                        # LOAD_1D_THERMAL.2 | name | list | case | type | value
                        # LOAD_1D_THERMAL.2 | name | item | case | type | pos_1 | value_1 | pos_2 | value_2
                        for el_ID in el_IDs:
                            ltype = 'CONS'
                            load_value = load_dict.get('DATA', 0)
                            ostr = [str(val) for val in ['LOAD_1D_THERMAL', '', el_ID, lc_ID, 
                                    ltype , load_value]]
                            gwa.write('\t'.join(ostr) + '\n')
            else:
                #print(f'el_ID: {el_ID} - no suitable elements found:')
                #print(f'   el_IDs: {el_IDs}, int_node_list: {int_node_list}')
                pass
            #pass

        # ========================
        # =====  Area Loads  =====
        # ========================
        if debug: print(f'\n===== Writing {len(AREA_LOAD_dict)} Area Loads to GWA =====')    
        
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
                    elif load_dict.get('TYPE') == 'TEMP':  # 'POINT' 
                        load_type = 'CONS' # Constant (uniform) temperature
                        load_value = load_dict.get('DATA', 0) #
                        # LOAD_2D_THERMAL.2 | name | list | case | type | values(n)
                        # LOAD_2D_FACE.2 | name | list | case | axis | type | proj | dir | value(n) | r | s
                        ostr = [str(val) for val in ['LOAD_2D_THERMAL.2', '', bm_max + sh_ID, lc_ID, 
                                load_type, load_value]]
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

        # ===============================
        # ========= EQ Spectra  =========
        # ===============================
        # Extracts and exports user-defined spectra
        # SPECTRUM.1	1	SPEC	USER	PERIOD	ACCEL	"(0,20.6206) (0.1,23.4851) (4.9,8.25021) (5,8.25021)"	CONST	CODE	0.05
        
        func_dict = E2K_dict.get('FUNCTIONS',{}).get('FUNCTION', {})
        spectrum_keys = [k for k, v in func_dict.items() if v.get('FUNCTYPE','') == 'SPECTRUM' and v.get('SPECTYPE','') == 'USER']
        
        if debug: print(f'\n===== Writing {len(spectrum_keys)} User Spectra to GWA =====')    
        
        for i, k in enumerate(spectrum_keys):
            damp = func_dict[k].get('DAMPRATIO', 0.0)
            damp_data = ['CONST', 'CODE', damp]
            time_val_list = func_dict[k].get('TIMEVAL', "")
            if len(time_val_list) > 0:
                data = ' '.join(time_val_list).split()
                timeacc_txt = '"' + ' '.join([f'({t},{grav * float(a)})' for t, a in zip(data[0::2], data[1::2])]) + '"'
                #print(f'SPECTRUM.1\t{i+1}\tSPEC\tUSER\tPERIOD\tACCEL\t' + txt)
                ostr = [str(val) for val in ['SPECTRUM.1', i+1, 'SPEC', 'USER', 'PERIOD', 'ACCEL'] + [timeacc_txt] + damp_data ]
                gwa.write('\t'.join(ostr) + '\n')




        if debug: 
            #print('\n**********************************************')
            print('\n***** Ending export of E2K_dict to GWA... *****')
            print('***********************************************')
    
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
    units = Units('N', 'm', 'C'),
    debug=False
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
        if debug:
            print('Write_GWA_model: Line list...')
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
                if debug: print(LINE_dict[id])

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
