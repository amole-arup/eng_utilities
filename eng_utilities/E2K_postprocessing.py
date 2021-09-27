""""""

from itertools import accumulate
from operator import itemgetter
from collections import namedtuple
from dateutil import parser
from os.path import exists, isfile, join, basename, splitext

from eng_utilities.general_utilities import try_numeric, unit_validate, Units
from eng_utilities.geometry_utilities import *
from eng_utilities.polyline_utilities import sec_area_3D
from eng_utilities.E2K_section_utilities import *
from eng_utilities.GWA_utilities import GWA_sec_gen, GWA_GEO
from eng_utilities.polyline_utilities import perim_full_props


Frame_Agg_Props = namedtuple('Frame_Agg_Props', 'material mat_type wt_density area')
Shell_Agg_Props = namedtuple('Shell_Agg_Props', 'material mat_type wt_density thickness')
Agg_Props = namedtuple('Agg_Props', 'material mat_type wt_density length area volume weight')




def enhance_frame_properties(f_name, f_dict, E2K_dict, 
                        section_def_dict, sec_key_dict, prop_file, 
                        model_units=Units('N', 'm', 'C')):
    """Add geometric properties to the frame props dictionary.
    
    Identify the type of section information provided 
       (e.g. catalogue, standard section such as Rectangular,
       SD Section, Embedded)
    """
    mat = f_dict.get('MATERIAL')
    if not mat:
        print(f'MATERIAL keyword is not present in f_dict: \n\t{f_dict}')
    MAT_PROP_dict = E2K_dict.get('MATERIAL PROPERTIES',{}).get('MATERIAL',{})
    if not MAT_PROP_dict:
        print(f'MAT_PROP_dict is missing')
    m_dict = MAT_PROP_dict.get(mat, {})
    if not m_dict:
        print(f'The material for {f_name} ({mat}) is not in MAT_PROP_dict')
        #raise ValueError('Missing material dictionary data')
    
    temp_f_dict_list = []
    res = None
    
    if f_dict['SHAPE'] == 'Auto Select List':
        pass
    elif f_dict['SHAPE'] == 'SD Section':
        pass  # this is addressed later
    elif 'Encasement' in f_dict['SHAPE']:
        enhance_encased_properties(f_dict, E2K_dict)
    elif f_dict['SHAPE'] == 'Nonprismatic':
        temp_f_dict_list.append(f_dict)  # delme after TODO
    elif (('FILE' in f_dict) or 
            (len(f_dict) < 4 and f_dict.get('ID')) or
            (len(f_dict) < 3 and (not f_dict.get('ID')))):   # a catalogue section
        stype = 'CAT'
        res = enhance_CAT_properties(f_dict, m_dict,  
                        section_def_dict, sec_key_dict, prop_file, 
                        model_units)
    else:
        stype = 'CALC' # assume it is a standard section
        f_dict['UNITS'] = E2K_dict.get('UNITS').length # add units for GWA string generation
        enhance_CALC_properties(f_dict, m_dict)
        
    # returns errors, all changes are made to the dictionary
    temp_f_dict_list.append(res)
    return temp_f_dict_list 


def enhance_encased_properties(f_dict, MAT_PROP_dict):
    """
    FRAMESECTION  "SRC_1200X1200"  MATERIAL "STEEL"  SHAPE "Concrete Encasement Rectangle"  D 1.200 B 1.200 ENCASEMENTMATERIAL "CONC" EMBEDDEDSECTION "700x700x25x32"
    """
    pass
    #if MAT_PROP_dict:
    #    m_dict = MAT_PROP_dict.get(f_dict.get('MATERIAL')

def enhance_SD_properties(f_dict, sd_dict):
    """Not Used"""
    agg_props_list = sd_dict.get('Frame_Agg_Props')
    if agg_props_list:
        f_dict['Frame_Agg_Props'] = agg_props_list


def enhance_CALC_properties(f_dict, m_dict):
    """"""
    shape = f_dict.get('SHAPE')
    
    if shape in ('Steel I/Wide Flange', 'I/Wide Flange', 'WIDE FLANGE'): 
        props = I_props_func(f_dict)
    elif shape in ('Steel Pipe', 'Concrete Pipe', 'Pipe', 'PIPE'): 
        props = CHS_props_func(f_dict)
    elif shape in ('Steel Tube', 'Concrete Tube', 'Tube', 'TUBE'):
        props = RH_props_func(f_dict)
    elif shape in ('Steel Angle', 'Concrete Angle', 'Angle'):
        props = A_props_func(f_dict)
    elif shape in ('Steel Double Angle', 'Concrete Double Angle', 'Double Angle'): 
        props = AA_props_func(f_dict)
    elif shape in ('Steel Plate', 'Concrete Rectangular', 'Rectangular', 'Rectangle', 'RECTANGLE'): 
        props = R_props_func(f_dict)
    elif shape in ('Steel Rod', 'Steel Circle', 'Concrete Circle', 'Circle', 'CIRCLE'): 
        props = C_props_func(f_dict)
    elif shape in ('Steel Channel', 'Concrete Channel', 'Channel', 'CHANNEL'): 
        props = CH_props_func(f_dict)
    else:
        print('Enhancement functions not found for:\n', f_dict)
        props = {}
        
    #('Concrete Encasement Rectangle',): CER_props_func,
    #('Concrete Encasement Circle',): CEC_props_func,
    #('Nonprismatic',): NP_props_func, 
    #('SD Section',): SD_props_func,

    #if not (isinstance(props, dict) and props):
    #    print(f'shape: {shape}, f_dict: {f_dict}')
    for k, v in props.items():
        f_dict[k] = v
    
    area = f_dict.get('A')
    mat = f_dict.get('MATERIAL')
    if mat and m_dict:
        wt_density = m_dict.get('W') if m_dict.get('W') else m_dict.get('WEIGHTPERVOLUME')
        mat_type = m_dict.get('DESIGNTYPE') if (m_dict.get('W') and m_dict.get('DESIGNTYPE')) else m_dict.get('TYPE')
    # 5. Place section properties into the dictionary (with units)
    if wt_density and area:
        f_dict['Frame_Agg_Props'] = [Frame_Agg_Props(mat, mat_type.casefold(), wt_density, area)]


def enhance_CAT_properties(f_dict, m_dict,  
                        section_def_dict, sec_key_dict, prop_file, 
                        model_units=Units('N', 'm', 'C')):
    """"""
    # For catalogue sections, the sections can be looked up.
    # but the units may need to be converted.
    # 1. Lookup correct file name using sec_key_dict lookup on lowercase name
    file_base = splitext(basename(f_dict.get('FILE', prop_file)))[0]
    prop_file = sec_key_dict[file_base.casefold()]
    f_dict['FILE'] = prop_file # update 'FILE' with useful propfile
    # 2. Lookup section properties
    shape_dict = get_cat_sec_props(f_dict, section_def_dict)
    if isinstance(shape_dict, dict):
        # 3. Place section properties into the dictionary (with units)
        for k, v in shape_dict.items():
            f_dict[k] = v
        
        # 4. Gather material density and converted section area
        sh_dict = convert_prop_units(shape_dict, model_units.length)
        area = sh_dict.get('A')
        mat = f_dict.get('MATERIAL')
        if mat and m_dict:
            wt_density = m_dict.get('W') if m_dict.get('W') else m_dict.get('WEIGHTPERVOLUME')
            mat_type = m_dict.get('DESIGNTYPE') if (m_dict.get('W') and m_dict.get('DESIGNTYPE')) else m_dict.get('TYPE')
        # 5. Place section properties into the dictionary (with units)
        if wt_density and area:
            f_dict['Frame_Agg_Props'] = [Frame_Agg_Props(mat, mat_type.casefold(), wt_density, area)]
        # 6. Add (likely) GWA descriptor string
        f_dict['GWA'] = GWA_sec_gen(f_dict)
    

def get_weight_density(m_dict):
    # WEIGHTPERVOLUME & TYPE
    # W & DESIGNTYPE (<= v9.7)
    return m_dict.get('W',0) if m_dict.get('W') else m_dict.get('WEIGHTPERVOLUME',0)


def get_mat_type(m_dict):
    # WEIGHTPERVOLUME & TYPE
    # W & DESIGNTYPE (<= v9.7)
    mtype = m_dict.get('DESIGNTYPE','') if m_dict.get('W') else m_dict.get('TYPE','')
    return mtype.casefold()


def enhance_shell_props(s_dict, MAT_PROP_dict):
    """"""
    if (s_dict.get('PROPTYPE').casefold() == 'deck'): # DECK
        conc_mat = s_dict.get('CONCMATERIAL')
        mc_dict = MAT_PROP_dict.get(conc_mat, {})
        mc_type = get_mat_type(mc_dict)
        mc_unit_wt = get_weight_density(mc_dict)
        deck_mat = s_dict.get('DECKMATERIAL')
        md_dict = MAT_PROP_dict.get(deck_mat, {})
        md_type = get_mat_type(md_dict)
        md_unit_wt = get_weight_density(md_dict)
        
        s_dict.update({k:v for k, v in deck_props_func(s_dict) if k in ['P', 'D_AVE']})
        B = s_dict.get('DECKRIBSPACING')
        P = s_dict.get('P')
        D_AVE = s_dict.get('D_AVE')
        T_AVE = s_dict.get('T_AVE')
        agg_props = []
        agg_props.append(Shell_Agg_Props(conc_mat, mc_type, mc_unit_wt, D_AVE))
        agg_props.append(Shell_Agg_Props(deck_mat, md_type, md_unit_wt, T_AVE))
        s_dict['Shell_Agg_Props'] = agg_props
        
    elif (s_dict.get('PROPTYPE').casefold() == 'wall'): # WALL
        conc_mat = s_dict.get('MATERIAL')
        mc_dict = MAT_PROP_dict.get(conc_mat, {})
        mc_type = get_mat_type(mc_dict)
        mc_unit_wt = get_weight_density(mc_dict)
        wall_thickness = s_dict.get('WALLTHICKNESS')
        agg_props = []
        agg_props.append(Shell_Agg_Props(conc_mat, mc_type.casefold(), mc_unit_wt, wall_thickness))
        s_dict['Shell_Agg_Props'] = agg_props
        
    elif (s_dict.get('PROPTYPE').casefold() == 'slab'): # SLAB
        conc_mat = s_dict.get('MATERIAL')
        mc_dict = MAT_PROP_dict.get(conc_mat, {})
        mc_type = get_mat_type(mc_dict)
        mc_unit_wt = get_weight_density(mc_dict)
        slab_thickness = s_dict.get('SLABTHICKNESS')
        agg_props = []
        agg_props.append(Shell_Agg_Props(conc_mat, mc_type.casefold(), mc_unit_wt, slab_thickness))
        s_dict['Shell_Agg_Props'] = agg_props



## =============================
## ===  E2K  PostProcessing  ===
## =============================

def get_E2K_subdict(the_dict, main_key, sub_key):
    """Returns the subdictionary specified by main_key
    and sub_key, returning an empty dictionary if any is missing.
    This is for use in the post-processing functions."""
    return the_dict[main_key].get(sub_key, {}) \
        if the_dict.get(main_key) else {}


def get_E2K_lookup_dict(the_dict, main_key, sub_key):
    """Returns a lookup dictionary specified by main_key
    and sub_key, returning an empty dictionary if any is missing.
    This is for use in the post-processing functions."""
    subdict = the_dict[main_key].get(sub_key, {}) \
        if the_dict.get(main_key) else {}
    if subdict:
        return {v.get('ID'): k for k, v in subdict.items() if v.get('ID')}


def FILE_PP(E2K_dict):
    """"""
    file_dict = E2K_dict.get('File')
    if file_dict:
        file_info = file_dict.get('Header')
        if '.$et saved ' in file_info: 
            ext = '.$et'
        elif '.$ET saved ' in file_info:
            ext = '.$ET'
        elif '.e2k saved ' in file_info:
            ext = '.e2k'
        elif '.E2K saved ' in file_info:
            ext = '.E2K'
        else:
            ext = ''

        file_path, date_txt = file_info.split(ext + ' saved ')
        file_path += ext

        file_date = parser.parse(date_txt)
        #print(file_path)
        #print(file_date)
        E2K_dict['FILEPATH'] = file_path
        E2K_dict['FILEDATE'] = file_date
        #return 0


def PROGRAM_PP(E2K_dict):
    """Postprocesses E2K_dict to extract program title and version
    """
    prog_dict = E2K_dict.get('PROGRAM INFORMATION')
    if isinstance(prog_dict, dict):
        prog_info = prog_dict.get('PROGRAM')
        prog_title = list(prog_info.keys())[0]
        prog_ver = prog_info[prog_title].get('VERSION')
        #print(f'{prog_title}: {prog_ver}')
        E2K_dict['PROGRAM_TITLE'] = prog_title
        E2K_dict['PROGRAM_VERSION'] = prog_ver


def CONTROLS_PP(E2K_dict):
    """Postprocesses E2K_dict to extract and standardise 
    units & titles
    
    TODO: could extract title information
    """
    control_dict = E2K_dict.get('CONTROLS')
    if control_dict:
        units_info = control_dict.get('UNITS')
        units = list(units_info.keys())[0]
        validated_units = [unit_validate(unit) for unit in units]
        if len(validated_units) == 2:
            temp_unit = 'C' if validated_units[1] in ('mm', 'dm', 'cm', 'm') else 'F'
            validated_units.append(temp_unit)
        E2K_dict['UNITS'] = Units(*validated_units)


def MAT_PROPERTIES_PP(E2K_dict):
    """Post-process properties for materials - add numerical IDs.
    """
    main_key = 'MATERIAL PROPERTIES'
    sub_key = 'MATERIAL'
    
    MAT_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    dict_keys  = MAT_PROP_dict.keys()
    
    for i, m_dict in enumerate(MAT_PROP_dict.values()):
        m_dict['ID'] = i + 1  # because GSA does not include number zero


def FRAME_SECTIONS_PP(E2K_dict, section_def_dict):
    """
    """
    prop_file_default = 'sections8' # for catalogue lookup
    temp_f_dict_list = [] # for logging examples
    sec_key_dict = {sec_name.casefold(): sec_name \
                    for sec_name in section_def_dict.keys()}
    prop_file = sec_key_dict[prop_file_default]
    
    print('Initially: ', E2K_dict.get('UNITS'))
    model_units = E2K_dict.get('UNITS', Units('N', 'm', 'C'))
    print('model_units: ', model_units)
    
    main_key = 'FRAME SECTIONS'
    sub_key = 'FRAMESECTION'
    
    FRAME_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    dict_keys  = FRAME_PROP_dict.keys()
    
    for i, (f_name, f_dict) in enumerate(FRAME_PROP_dict.items()):
        f_dict['ID'] = i + 1  # because GSA does not include number zero
        #print('f_dict: ', f_dict)
        
        if f_dict.get('FILE'): # check format and correct if necessary
            # Older files have filepath, so convert these to file base-name
            file_base = splitext(basename(f_dict.get('FILE', prop_file)))[0]
            prop_file = sec_key_dict[file_base.casefold()]
            f_dict['FILE'] = prop_file
        
        if f_dict.get('A'):
            res = None  # properties have already been enhanced        
        else:
            res = enhance_frame_properties(f_name, f_dict, E2K_dict, section_def_dict, sec_key_dict, prop_file, model_units)
        
        # update default prop_file (since this becomes the default)
        if f_dict.get('FILE'):
            prop_file = f_dict.get('FILE')
        #temp_f_dict_list.append(res) ## delme after TODO
    #print('\n', temp_f_dict_list) ## delme after TODO


def ENCASED_SECTIONS_PP(E2K_dict):
    """Post-process properties for encased sections
    """
    main_key = 'MATERIAL PROPERTIES'
    sub_key = 'MATERIAL'
    MAT_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    main_key = 'FRAME SECTIONS'
    sub_key = 'FRAMESECTION'
    FRAME_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    for f_name, f_dict in FRAME_PROP_dict.items():
        if 'Encasement' in f_dict['SHAPE']:
            embed_sect = f_dict.get('EMBEDDEDSECTION')
            embed_mat = f_dict.get('MATERIAL')
            embed_m_dict = MAT_PROP_dict.get(embed_mat, {})
            embed_props = FRAME_PROP_dict.get(embed_sect)
            embed_unit_wt = embed_m_dict.get('W') \
                            if embed_m_dict.get('W') \
                            else embed_m_dict.get('WEIGHTPERVOLUME')
            embed_mat_type = embed_m_dict.get('DESIGNTYPE') if (embed_m_dict.get('W') and embed_m_dict.get('DESIGNTYPE')) else embed_m_dict.get('TYPE')
            
            encase_mat = f_dict.get('ENCASEMENTMATERIAL')
            encase_m_dict = MAT_PROP_dict.get(encase_mat, {})
            encase_unit_wt = encase_m_dict.get('W') \
                            if encase_m_dict.get('W') \
                            else encase_m_dict.get('WEIGHTPERVOLUME')
            encase_mat_type = encase_m_dict.get('DESIGNTYPE') if (encase_m_dict.get('W') and encase_m_dict.get('DESIGNTYPE')) else encase_m_dict.get('TYPE')
            
            if f_dict['SHAPE'].endswith('Rectangle'):
                encase_props = R_props_func(f_dict)
                print('encase_props (R)', encase_props)
            elif f_dict['SHAPE'].endswith('Circle'):
                encase_props = C_props_func(f_dict)
                print('encase_props (C)', encase_props)
            
            agg_props = []
            embed_area = embed_props.get('A')
            if embed_props.get('UNITS'):
                model_units = E2K_dict('UNITS')
                embed_props = convert_prop_units(embed_props, model_units.length)
                embed_area = embed_props.get('A')
            agg_props.append(Frame_Agg_Props(embed_mat, embed_mat_type.casefold(), embed_unit_wt, embed_area))
            
            encase_area = encase_props.get('A') - embed_area
            agg_props.append(Frame_Agg_Props(encase_mat, encase_mat_type.casefold(), encase_unit_wt, encase_area))
            
            f_dict['Frame_Agg_Props'] = agg_props


def NONPRISMATIC_SECTIONS_PP(E2K_dict):
    """"""
    pass


def SD_SECTIONS_PP(E2K_dict):
    """
    Post-process section designer data - non-standard sections
    
    Note that this should take place after processing the frame sections and will add data into the E2K_dict['FRAME SECTIONS'] dictionary
    Note also that if other SHAPETYPES are processed, 
    it will be necessary to do some subtraction...
    """
    
    main_key = 'MATERIAL PROPERTIES'
    sub_key = 'MATERIAL'
    MAT_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    main_key = 'SECTION DESIGNER SECTIONS'
    sub_key = 'SDSECTION'
    SD_SECTION_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    units = E2K_dict.get('UNITS')
    
    for sd_sect, sd_data in SD_SECTION_dict.items():
        shapes = sd_data.get('SHAPE')
        mat_area_list = []
        if shapes:
            for shape in shapes.values():
                # add other options for SHAPETYPES, such as custom embedded sections
                if shape.get('MATERIAL') and shape.get('SHAPETYPE') == 'POLYGON':
                    polyline = list(zip(shape['X'], shape['Y']))
                    poly_props = perim_full_props(polyline)
                    area = poly_props['A'] # perim_area_centroid(polyline)[0]
                    perimeter = poly_props['P']
                    GWA = GWA_GEO(polyline, units=units.length)
                    sd_data.update({'A': area, 'P': perimeter, 'GWA': GWA})
                    
                    unit_weight = None
                    mat = shape['MATERIAL']
                    mdict = MAT_PROP_dict.get(mat)
                    if mdict:
                        unit_weight = mdict.get('WEIGHTPERVOLUME') if mdict.get('WEIGHTPERVOLUME') else mdict.get('W',0)
                        mat_type = mdict.get('DESIGNTYPE') if (mdict.get('W') and mdict.get('DESIGNTYPE')) else mdict.get('TYPE')
                    mat_area_list.append(Frame_Agg_Props(mat, mat_type.casefold(), unit_weight, area))
                elif shape.get('SHAPETYPE') == 'POLYGON':  
                    polyline = list(zip(shape['X'], shape['Y']))
                    poly_props = perim_full_props(polyline)
                    area = poly_props['A']
                    perimeter = poly_props['P']
                    GWA = GWA_GEO(polyline, units=units.length)
                    sd_data.update({'A': area, 'P': perimeter, 'GWA': GWA})
                    
                else:
                    # add other options for custom embedded sections
                    pass
        #print('sd_data: ', sd_data)
        sd_data['Frame_Agg_Props'] = mat_area_list
        #SD_SECTION_dict[sd_sect]['Frame_Agg_Props'] = mat_area_list
    
    
    # Insert SD data into FRAME PROPERTIES dictionary
    main_key = 'FRAME SECTIONS'
    sub_key = 'FRAMESECTION'
    FRAME_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    for f_name, f_dict in FRAME_PROP_dict.items():
        if f_dict['SHAPE'] == 'SD Section':
            f_dict.update(SD_SECTION_dict.get(f_name, {}))
            #sd_sect = SD_SECTION_dict.get(f_name, {})
            #if sd_sect.get('Frame_Agg_Props'): 
            #    f_dict['Frame_Agg_Props'] = sd_sect['Frame_Agg_Props']


def SHELL_PROPERTIES_PP(E2K_dict):
    """Post-process shell properties - adding numerical IDs and aggregate properties for quantities
    
    TODO: 
        1. Need to handle case in PROPTYPE - older versions use 'all caps'.
        2. Need to generate unit area aggregated properties (thickness, 
            weight per unit area etc). The element type needs to be carried
    
    Shell_Agg_Props = namedtuple('Shell_Agg_Props', 'material mat_type wt_density thickness')
    """
    main_key = 'MATERIAL PROPERTIES'
    sub_key = 'MATERIAL'
    MAT_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    # For older versions of ETABS
    main_key = 'WALL/SLAB/DECK PROPERTIES'
    sub_key = 'SHELLPROP'
    SHELL_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    if not SHELL_PROP_dict:
        # Combine Shell Dictionaries - Slab, Deck, Wall
        SLAB_PROP_dict = get_E2K_subdict(E2K_dict, 'SLAB PROPERTIES', 'SHELLPROP')
        DECK_PROP_dict = get_E2K_subdict(E2K_dict, 'DECK PROPERTIES', 'SHELLPROP')
        WALL_PROP_dict = get_E2K_subdict(E2K_dict, 'WALL PROPERTIES', 'SHELLPROP')
        
        SHELL_PROP_dict = {**SLAB_PROP_dict, **DECK_PROP_dict, **WALL_PROP_dict}
    
    # Enhance the properties with aggregated values
    for i, s_dict in enumerate(SHELL_PROP_dict.values()):
        s_dict['ID'] = i + 1  # because GSA does not include number zero
        enhance_shell_props(s_dict, MAT_PROP_dict)
    
    E2K_dict['SHELL PROPERTIES'] = {'SHELLPROP': SHELL_PROP_dict}


def STORIES_PP(E2K_dict):
    """Postprocesses E2K_dict to add elevations to story data
    
    TODO: this will need to be revised to take 
    'Tower' into account
    """
    if E2K_dict.get('STORIES - IN SEQUENCE FROM TOP'):
        STORY_dict = E2K_dict['STORIES - IN SEQUENCE FROM TOP'].get('STORY')
        if STORY_dict:
            STORY_keys = STORY_dict.keys()
            base = sum(STORY_dict[key].get('ELEV',0) for key in STORY_keys)
            heights = [STORY_dict[key].get('HEIGHT',0) for key in STORY_keys]
            relative_elevations = list(accumulate(heights[::-1]))[::-1]
            absolute_elevations = [base + relev for relev in relative_elevations]
            for key, relev, abs_elev in zip(STORY_keys, relative_elevations, absolute_elevations):
                STORY_dict[key]['RELEV'] = relev
                STORY_dict[key]['ABS_ELEV'] = abs_elev    


def POINTS_PP(E2K_dict):
    """'POINT COORDINATES': Postprocesses E2K_dict to organise 
    points, coords into key, value pairs if they are not already.
    
    Dictionary Approach - note that coordinates are lumped 
    together in tuples of (X, Y, DeltaZ)
    """
    main_key = 'POINT COORDINATES'
    sub_key = main_key.split()[0]
    
    #POINTS_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    POINTS_dict = E2K_dict['POINT COORDINATES']['POINT']
    point_keys = list(POINTS_dict.keys()) #
    if isinstance(point_keys[0], (tuple, list)):
        POINTS_dict = {try_numeric(pt[0]): pt[1:] for pt, val in POINTS_dict.items() if val == dict()}
        # Need to reassign data back to E2K_dict
        E2K_dict['POINT COORDINATES']['POINT'] = POINTS_dict


def POINT_ASSIGNS_PP(E2K_dict):
    """'POINT ASSIGNS': Postprocesses E2K_dict to add
    coordinates to every node.
    
    NOTE: not all points are present in this dictionary, so 
    additional values will be added as necessary by other 
    post-processing in the element assignations
    """
    # Get reference to story elevations
    main_key = 'STORIES - IN SEQUENCE FROM TOP'
    sub_key = 'STORY'
    STORY_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    # Get reference to points coordinates
    main_key = 'POINT COORDINATES'
    sub_key = main_key.split()[0]
    POINTS_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    # Get reference to Point Assignments
    main_key = 'POINT ASSIGNS'
    sub_key = main_key[:-1].replace(' ','')
    NODES_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    dict_keys  = NODES_dict.keys()
    
    # Check if dictionary has already been processed
    # If it has, we don't want this messing with the IDs
    if not NODES_dict[list(dict_keys)[0]].get('ID'):
        for i, key in enumerate(dict_keys):
            point, story = key
            coords = POINTS_dict[point]
            if len(coords) == 3:
                x, y, dz = coords
                NODES_dict[key]['DELTAZ'] = dz
            else:
                x, y = coords
                dz = 0
            abselev = STORY_dict[story]['ABS_ELEV']
            #abselev = POINTS_dict.get(point, None)
            
            NODES_dict[key]['COORDS'] = (x, y, abselev - dz)
            NODES_dict[key]['ID'] = i + 1


def LINE_CONN_PP(E2K_dict):
    """'LINE CONNECTIVITIES': Postprocesses E2K_dict to 
    extract element type and organise connection data.
    
    """
    main_key = 'LINE CONNECTIVITIES'
    sub_key = main_key.split()[0]

    if E2K_dict.get(main_key):
        if E2K_dict[main_key].get(sub_key):
            LINES_dict = E2K_dict[main_key][sub_key]

            for k, v in LINES_dict.items():
                if not v.get('Type'):
                    pd_list = []
                    for k2,v2 in v.items():
                        pd_list.append(('Type', k2))
                        pd_list.append(('N1', (v2[0],v2[2])))
                        pd_list.append(('N2', (v2[1],0)))
                    #print(pd_list)
                    for k3, v3 in pd_list:
                        LINES_dict[k][k3] = v3


def LINE_ASSIGNS_PP(E2K_dict):
    """'LINE ASSIGNS': Postprocesses E2K_dict to add
    coordinates, lengths, areas, volumes and weights 
    to every line assignment.
    
    NOTE: not all points are present in the NODES dictionary, 
    so additional values will be added to NODES_dict as
    necessary.
    """
    my_log = []
    
    # Get reference to story elevations
    main_key = 'STORIES - IN SEQUENCE FROM TOP'
    sub_key = 'STORY'
    STORY_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    # STORY_lookup - provide index, get story
    story_flag = False
    if STORY_dict:
        STORY_keys = STORY_dict.keys()
        STORY_lookup = {i+1: story for i, story in enumerate(list(STORY_keys)[::-1])}
        
        # STORY_reverse_lookup - provide story, get index
        STORY_reverse_lookup = {v:k for k, v in STORY_lookup.items()}
        story_flag = True
    
    # Get reference to points (only required if point is not referenced
    # in the POINT ASSIGN dictionary)
    main_key = 'POINT COORDINATES'
    sub_key = 'POINT'
    POINTS_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    # Check NODES_dict for adding new nodes to NODES_dict
    nodes_flag = False
    NODES_dict = E2K_dict.get('POINT ASSIGNS', {}).get('POINTASSIGN', {})
    if NODES_dict:
        node_max_orig = len(NODES_dict)
        next_node_id = len(NODES_dict) + 1  ###  NB ID is 1-based not 0-based  ###
        nodes_flag = True
    
    # Get reference to sections 
    main_key = 'FRAME SECTIONS'
    sub_key = 'FRAMESECTION'
    FRAME_PROP_dict = E2K_dict.get(main_key, {}).get(sub_key, {})
    
    # Check LINES_dict that is to be referenced
    LINES_dict = E2K_dict.get('LINE CONNECTIVITIES', {}).get('LINE', {})

    main_key = 'LINE ASSIGNS'
    sub_key = main_key[:-1].replace(' ','')
    
    # Consolidate all the checks
    all_OK = False
    if E2K_dict.get(main_key) and story_flag and nodes_flag and LINES_dict:
        if E2K_dict[main_key].get(sub_key):
            MEMBERS_dict = E2K_dict[main_key][sub_key]
            MEMBERS_keys  = MEMBERS_dict.keys()

            # Check if dictionary has already been processed
            # If it has, we don't want this messing with the IDs
            if not MEMBERS_dict[list(MEMBERS_keys)[0]].get('ID'):
                all_OK = True
    
    if all_OK:    
        # Lookup Node_1 & Node_2 and convert into NODE references
        for i, key in enumerate(MEMBERS_keys):
            #if i<3: print(f'i: {i} | key: {key}')
            line, story = key
            line_dict = LINES_dict.get(line)
            MEMBERS_dict[key]['ID'] = i + 1    ###  NB ID is 1-based not 0-based  ###
            story_index = STORY_reverse_lookup[story]
            line_pts = []
            MEMBERS_dict[key]['MEMTYPE'] = line_dict.get('Type')
            
            for n in ('1', '2'):
                point_n, drop_n = line_dict.get('N' + n)
                if drop_n == 0:
                    story_n = story
                else:
                    story_n = STORY_lookup.get(story_index - drop_n)
                MEMBERS_dict[key]['JT' + n] = (point_n, story_n)
                
                ndict = NODES_dict.get((point_n, story_n))
                if isinstance(ndict,dict):
                    MEMBERS_dict[key]['N' + n] = int(ndict.get('ID'))
                    coords_n = ndict.get('COORDS')
                    MEMBERS_dict[key]['COORDS' + n] = coords_n
                    line_pts.append(coords_n)
                else:
                    coords_rel = POINTS_dict.get(point_n, None)
                    sdict = STORY_dict.get(story_n, None)
                    if coords_rel and sdict:
                        deltaZ = 0 if (len(coords_rel) < 3) else coords_rel[2]
                        Z = sdict.get('ABS_ELEV') - deltaZ
                        coords_n = (coords_rel[0], coords_rel[1], Z)
                        NODES_dict[(point_n, story_n)] = {'ID': next_node_id, 'COORDS':coords_n}
                        MEMBERS_dict[key]['N' + n] = next_node_id
                        MEMBERS_dict[key]['COORDS' + n] = coords_n
                        line_pts.append(coords_n)
                        next_node_id += 1
                    else:
                        my_log.append(f'LINE ASSIGNS: Node lookup failed for {key} at N{n}: {(point_n, story_n)}')
            
            # add member length
            length = None
            clear_length = None
            if len(line_pts) == 2:
                length = dist3D(line_pts[0], line_pts[1])
                clear_length = length - MEMBERS_dict[key].get('LENGTHOFFI', 0) - MEMBERS_dict[key].get('LENGTHOFFJ', 0)
                MEMBERS_dict[key]['L'] = length
                MEMBERS_dict[key]['L_c'] = clear_length
            
            # add section area (needs access to section definition containing section areas etc
            f_dict = FRAME_PROP_dict.get(MEMBERS_dict[key].get('SECTION'),{})
            agg_props = f_dict.get('Frame_Agg_Props', [])
            agg_props2 = []
            
            propmod_w = MEMBERS_dict[key].get('PROPMODW', 1)
            if clear_length:
                for agg_prop in agg_props:
                    agg_props2.append(Agg_Props(
                        agg_prop.material, 
                        agg_prop.mat_type, 
                        agg_prop.wt_density,
                        clear_length, 
                        agg_prop.area, 
                        agg_prop.area * clear_length, 
                        agg_prop.area * clear_length * agg_prop.wt_density * propmod_w))
                MEMBERS_dict[key]['Memb_Agg_Props'] = agg_props2
    
    
    if True:    
        ## Debugging CHECKS ##
        node_max_new = len(NODES_dict)
        node_max_change = node_max_new - node_max_orig
        print(f'Number of nodes has changed from {node_max_orig} to {node_max_new}')
        print(f'    a change of {node_max_change}\n')

        print(f'Number of errors: {len(my_log)}')
        print(f'Number of members: {len(MEMBERS_dict)}\n')
        print(my_log)


def AREA_CONN_PP(E2K_dict):
    """'AREA CONNECTIVITIES': Postprocesses E2K_dict to 
    extract element type and organise connection data.
    
    """
    main_key = 'AREA CONNECTIVITIES'
    sub_key = main_key.split()[0]

    if E2K_dict.get(main_key):
        if E2K_dict[main_key].get(sub_key):
            AREAS_dict = E2K_dict[main_key][sub_key]

            for k, v in AREAS_dict.items():
                if not v.get('Type'):
                    pd_list = []
                    for k2,v2 in v.items():
                        pd_list.append(('Type', k2))
                        pd_list.append(('Num', v2[0]))
                        pd_list.extend([('N'+str(i+1),(a, b)) 
                                    for i, (a, b) in enumerate(zip(v2[1:(len(v2)//2+1)], 
                                                                   v2[(len(v2)//2+1):]))])
                        
                    #print(pd_list)
                    for k3, v3 in pd_list:
                        AREAS_dict[k][k3] = v3


def AREA_ASSIGNS_PP(E2K_dict):
    """'AREA ASSIGNS': Postprocesses E2K_dict to add
    coordinates, thicknesses, areas, volumes and weights 
    to every line assignment.
    
    NOTE: not all points are present in the NODES dictionary, 
    so additional values will be added to NODES_dict as
    necessary.
    """
    my_log = []
    

    ## == main processes == ##
    main_key = 'AREA ASSIGNS'
    sub_key = main_key[:-1].replace(' ','')
    SHELLS_dict = E2K_dict.get(main_key,{}).get(sub_key,{})

    if not SHELLS_dict:
        return

    # Area Connectivities
    AREAS_dict = E2K_dict('AREA CONNECTIVITIES',{}).get('AREA',{})
    
    # Get reference to story elevations
    main_key = 'STORIES - IN SEQUENCE FROM TOP'
    sub_key = 'STORY'
    STORY_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    # STORY_lookup - provide index, get story
    story_flag = False
    if STORY_dict:
        STORY_keys = STORY_dict.keys()
        STORY_lookup = {i+1: story for i, story in enumerate(list(STORY_keys)[::-1])}
        
        # STORY_reverse_lookup - provide story, get index
        STORY_reverse_lookup = {v:k for k, v in STORY_lookup.items()}
        story_flag = True
    
    # Get reference to points (only required if point is not referenced
    # in the POINT ASSIGN dictionary)
    main_key = 'POINT COORDINATES'
    sub_key = 'POINT'
    POINTS_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    
    # Check NODES_dict for adding new nodes to NODES_dict
    nodes_flag = False
    NODES_dict = E2K_dict.get('POINT ASSIGNS', {}).get('POINTASSIGN', {})
    if NODES_dict:
        node_max_orig = len(NODES_dict)
        next_node_id = len(NODES_dict) + 1  ###  NB ID is 1-based not 0-based  ###
        nodes_flag = True

    # Get reference to sections 
    main_key = 'SHELL PROPERTIES'
    sub_key = 'SHELLPROP'
    SHELL_PROP_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)    

    all_OK = False # Final checks
    if story_flag and nodes_flag and AREAS_dict:
        SHELLS_keys  = SHELLS_dict.keys()

        # Check if dictionary has already been processed
        # If it has, we don't want this messing with the IDs
        if not SHELLS_dict[list(SHELLS_keys)[0]].get('ID'):
            all_OK = True
    
    if all_OK:    
        # Lookup all nodes (1, 2, 3, etc) and convert into NODE references
        for i, key in enumerate(SHELLS_keys):
            area, story = key
            area_dict = AREAS_dict.get(area)
            num_pts = area_dict.get('Num')
            SHELLS_dict[key]['ID'] = i + 1    ###  NB ID is 1-based not 0-based  ###
            story_index = STORY_reverse_lookup[story]
            area_pts = []
            Nn_list = [str(i+1) for i in range(num_pts)]
            SHELLS_dict[key]['NumPts'] = num_pts
            SHELLS_dict[key]['MEMTYPE'] = area_dict.get('Type')
            
            for n in Nn_list:
                point_n, drop_n = area_dict.get('N' + n)
                if drop_n == 0:
                    story_n = story
                else:
                    story_n = STORY_lookup.get(story_index - drop_n)
                SHELLS_dict[key]['JT' + n] = (point_n, story_n)
                
                ndict = NODES_dict.get((point_n, story_n))
                if isinstance(ndict,dict):
                    SHELLS_dict[key]['N' + n] = int(ndict.get('ID'))
                    coords_n = ndict.get('COORDS')
                    SHELLS_dict[key]['COORDS' + n] = coords_n
                    area_pts.append(coords_n)
                else:
                    
                    coords_rel = POINTS_dict.get(point_n, None)
                    sdict = STORY_dict.get(story_n, None)
                    if coords_rel and sdict:
                        deltaZ = 0 if (len(coords_rel) < 3) else coords_rel[2]
                        Z = sdict.get('ABS_ELEV') - deltaZ
                        coords_n = (coords_rel[0], coords_rel[1], Z)
                        NODES_dict[(point_n, story_n)] = {'ID': next_node_id, 'COORDS':coords_n}
                        SHELLS_dict[key]['N' + n] = next_node_id
                        SHELLS_dict[key]['COORDS' + n] = coords_n
                        area_pts.append(coords_n)
                        next_node_id += 1
                    else:
                        my_log.append(f'AREA ASSIGNS: Node lookup failed for {key} at N{n}: {(point_n, story_n)}')
                
            # add member area
            shell_area = 0
            if len(area_pts) > 2:
                shell_area = sec_area_3D(area_pts)
                SHELLS_dict[key]['A'] = shell_area
            
            # add section thickness (needs access to section definition containing section areas etc
            s_dict = SHELL_PROP_dict.get(SHELLS_dict[key].get('SECTION'),{})
            agg_props = s_dict.get('Shell_Agg_Props', [])
            agg_props2 = []
            propmod_w = SHELLS_dict[key].get('PROPMODW', 1)
            if shell_area:
                for agg_prop in agg_props:
                    agg_props2.append(Agg_Props(
                        agg_prop.material, 
                        agg_prop.mat_type, 
                        agg_prop.wt_density, 
                        agg_prop.thickness, 
                        shell_area, 
                        agg_prop.thickness * shell_area, 
                        agg_prop.thickness * shell_area * agg_prop.wt_density * propmod_w))
                SHELLS_dict[key]['Memb_Agg_Props'] = agg_props2
        
        
    if True:
        ## Debugging CHECKS ##
        node_max_new = len(NODES_dict)
        node_max_change = node_max_new - node_max_orig
        print(f'Number of nodes has changed from {node_max_orig} to {node_max_new}')
        print(f'    a change of {node_max_change}\n')

        print(f'Number of errors: {len(my_log)}')
        print(f'Number of shells: {len(SHELLS_dict)}\n')
        print(my_log)


def LOAD_CASES_PP(E2K_dict):
    """'LOADCASE' or 'LOADPATTERN' : Postprocesses E2K_dict to 
    add a load case / load pattern integer ID 
    to every load case / load pattern definition.
    """
    #my_log = []
    

    ## == main processes == ##
    
    STATIC_LOAD_dict = E2K_dict.get('STATIC LOADS',{}).get('LOADCASE',{})
    LOAD_PATTERNS_dict = E2K_dict.get('LOAD PATTERNS',{}).get('LOADPATTERN',{})

    if STATIC_LOAD_dict:
        the_dict = STATIC_LOAD_dict
    else:
        the_dict = LOAD_PATTERNS_dict
    
    for i, lc_dict in enumerate(the_dict.values()):
        lc_dict['ID'] = i + 1
        # return



## =============================
## ===  Quantities Take-off  ===
## =============================

def quantities_summary(M_dict):
    """Pulls out the materials quantities data and 
    creates a general summary with keys: 
        story, member type, material type, material name
    
    display(pd.DataFrame.from_dict(quantities_summary(MEMBERS_dict), orient='index'))"""
    dlist = []
    [[dlist.append(((k[1], k[0], data.get('MEMTYPE'), 
                     datum.mat_type, datum.material), datum)) 
      for datum in data.get('Memb_Agg_Props')] 
     for k, data in M_dict.items()]
    
    #levels = set([k[0] for k, _ in dlist])
    #mem_types = set([str(k[1])[0] for k, _ in dlist])
    #mem_class = set([k[2] for k, _ in dlist])
    #mats = set([k[3] for k, _ in dlist])
    #mat_names = set([k[4] for k, _ in dlist])
    #display(f'levels, {levels}, mem_types, {mem_types}, ' + \
    #        f'mem_class, {mem_class}, mats, {mats}, mat_names, {mat_names}')
    
    #print()
    #print('D_LIST')
    #print(dlist[:20],'\n')

    sum_dict = dict()
    # Carry out the summations
    for i, (k, v) in enumerate(dlist):
        #if i<1: print(f'k: {k}, v: {v}')
        key = k[0], k[2], k[3], k[4]
        rdict = sum_dict.get(key, {})
        if rdict:
            for d, w in zip(['length', 'area', 'volume', 'weight'], 
                            [v.length, v.area, v.volume, v.weight]):
                t = rdict[d]
                rdict[d] = t + w
        else:
            rdict = {'length':0, 'area':0, 'volume':0, 'weight':0}
        sum_dict[key] = rdict
    return sum_dict


def MEMBER_quantities_summary(E2K_dict, descending = True):
    """Extracts the materials quantities and places them into 
    dictionaries inside the main dictionary with keys: 
        (story, member type, material type, material name)    
    """
    MEMBERS_dict = E2K_dict.get('LINE ASSIGNS', {}).get('LINEASSIGN', {})
    SHELLS_dict = E2K_dict.get('AREA ASSIGNS', {}).get('AREAASSIGN', {})
    for name, M_dict in (('MEMBERS SUMMARY', MEMBERS_dict), ('SHELLS SUMMARY', SHELLS_dict)):
        sum_dict = quantities_summary(M_dict)
        STORY_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('STORY', {})
        if descending:
            story_order = STORY_dict.keys()
        else:
            story_order = list(STORY_dict.keys())[::-1]
        
        sum_dict_keys = sum_dict.keys()
        story_ordered = [[k for k in sum_dict_keys if k[0] == s_key] for s_key in story_order]
        [one_list.sort(key = itemgetter(1, 2, 3)) for one_list in story_ordered]
        E2K_dict[name] = {k:sum_dict.get(k) for k in sum(story_ordered,[])}
        
        #d2list = []
        #[[d2list.append(k) for k in sum_dict.keys() if k[0] == k1] for k1 in story_order]
        #if descending:
            #E2K_dict[name] = {k:sum_dict.get(k) for k in d2list}
        #else:
        #    E2K_dict[name] = {k:sum_dict.get(k) for k in d2list[::-1]}


def MODEL_quantities_summary(E2K_dict, descending = True):
    """Extracts the materials quantities and places them into 
    dictionaries inside the main dictionary with keys: 
        (story, member type, material type, material name)    
    """
    # TODO calculate summaries in the Notebook...
    MEMBERS_qdict = E2K_dict.get('MEMBERS SUMMARY', {})
    SHELLS_qdict = E2K_dict.get('SHELLS SUMMARY', {})

