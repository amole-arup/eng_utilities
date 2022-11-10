"""Contains post-processing operations for gathering section quantities 
and also, in some cases, prepares for GWA export.

TODO: 
* Add logic to SD_SECTIONS_PP for when it is not simply one Polygon
* Work out what USERJOINTS are and uncomment the relevant code that has already been written
"""

from itertools import accumulate
from operator import itemgetter
from collections import namedtuple
from dateutil import parser
from os.path import exists, isfile, join, basename, splitext

from eng_utilities.general_utilities import try_numeric, unit_validate, Units, units_conversion_factor, rounder
from eng_utilities.geometry_utilities import *
from eng_utilities.polyline_utilities import sec_area_3D, perim_area_centroid, line_intersection2D
from eng_utilities.E2K_section_utilities import *
from eng_utilities.GWA_utilities import GWA_sec_gen, GWA_GEO
from eng_utilities.polyline_utilities import perim_full_props, all_loops_finder


Frame_Agg_Props = namedtuple('Frame_Agg_Props', 'material mat_type wt_density area')
Shell_Agg_Props = namedtuple('Shell_Agg_Props', 'material mat_type wt_density thickness')
Agg_Props = namedtuple('Agg_Props', 'material mat_type wt_density length area volume weight')


def rel_story_alt(story_name, Story_List_dict, n_down, tag='', debug=False):
    """Returns the story relative to the story provided

    NB There are problems with multi-tower models since n_down often
    does not seem to correspond to any real drop

    Args:
        story_name (str, int): this is the story name as used (i.e. including 
            the tower name if present - typically joined with a hyphen, 
            e.g. 'TowerOne-L22')
        Story_List_dict - the 'Story_Lists' dictionary in the E2K_dict, 
            e.g. Story_List_dict = E2K_dict['STORIES - IN SEQUENCE FROM TOP']['Story_Lists']
            Note that for buildings without 'Towers' there is a 'Default' tower.
        n_down (int): the number of stories to descend. It automatically cuts off
            above and below (will not go up above roof or down below base)
    """
    # If story_name contains the name of a tower, then the following will be
    #   a double-length list comprising the Tower and the Story name 
    split_story_name = story_name.split('-', 1) if isinstance(story_name, str) else [story_name]
    
    # Check where story name is in Story Lists
    
    #   Check against 'Default' Story_List
    if 'Default' in Story_List_dict.keys():
        story_list = list(Story_List_dict.get('Default', []))
        if story_name in story_list:
            story_index = story_list.index(story_name)
        else:
            if len(split_story_name) == 2 and split_story_name[-1] in story_list:
                story_index = story_list.index(story_name)
            else:
                story_index = None
    elif len(split_story_name) > 1:
        story_list = list(Story_List_dict.get(split_story_name[0], []))
        story_index = story_list.index(story_name)
    else:
        towers = [k for k, v in Story_List_dict.items() if story_name in v]
        
        if len(towers) == 1:  # If there is only one match
            story_list = list(Story_List_dict.get(towers[0], []))
            story_index = story_list.index(story_name)
        else:
            story_index = None
            story_list = []

    # if characters before the hyphen correspond to a "tower" 
    #     this will return a list of storeys
    # if the story is numeric, or anything other than a string, 
    #     it will be returned as one item in a list
    
    if debug and story_index is None:
        err_tag = f' [{tag}]' if tag else ''
        print(f'%% Story Lookup Error - going down {n_down} from {story_name} (storey_index is None)' + err_tag)
        print(f'story_list is: {story_list}')
        print()
    
    return story_list[max(0, min(story_index + n_down, len(story_list) - 1))]


def rel_story(story_name, Story_List_dict, n_down, tag='', debug=False):
    """Returns the story relative to the story provided

    NB There are problems with multi-tower models since n_down often
    does not seem to correspond to any real drop

    Args:
        story_name (str, int): this is the story name as used (i.e. including 
            the tower name if present - typically joined with a hyphen, 
            e.g. 'TowerOne-L22')
        Story_List_dict - the 'Story_Lists' dictionary in the E2K_dict, 
            e.g. Story_List_dict = E2K_dict['STORIES - IN SEQUENCE FROM TOP']['Story_Lists']
            Note that for buildings without 'Towers' there is a 'Default' tower.
        n_down (int): the number of stories to descend. It automatically cuts off
            above and below (will not go up above roof or down below base)
    
    """
    # if characters before the hyphen correspond to a "tower" 
    # this will return a list of storeys
    story_name_split = story_name.split('-', 1) if isinstance(story_name, str) else [story_name]
    story_list = Story_List_dict.get(story_name_split[0], [])
    
    if (len(story_name_split) == 0) or len(story_list) == 0:
        story_list = list(Story_List_dict.get('Default', []))
    
    story_index = story_list.index(story_name) if story_name in story_list else None
    
    if debug and story_index is None:
        err_tag = f' [{tag}]' if tag else ''
        print(f'%% Story Lookup Error - going down {n_down} from {story_name}' + err_tag)
        print()
    
    return story_list[max(0,min(story_index + n_down, len(story_list) - 1))]



def enhance_frame_properties(f_name, f_dict, E2K_dict, 
                        section_def_dict, sec_key_dict, prop_file, 
                        model_units=Units('N', 'm', 'C'), debug=False):
    """Add geometric properties to the frame props dictionary.
    
    Identify the type of section information provided 
       (e.g. catalogue, standard section such as Rectangular,
       SD Section, Embedded)
    """
    MAT_PROP_dict = E2K_dict.get('MATERIAL PROPERTIES',{}).get('MATERIAL',{})
    # if debug and (not MAT_PROP_dict): print(f'MAT_PROP_dict is missing')

    mat = f_dict.get('MATERIAL', None)
    shape = f_dict.get('SHAPE', None)
    m_dict = MAT_PROP_dict.get(mat, MAT_PROP_dict.get(mat.casefold(), {})) if mat is not None else {}
    
    if mat is None:
        # logging errors
        if (shape.casefold() != 'nonprismatic'):
            if debug: print(f'Log: MATERIAL keyword is not present in {f_name} dict: \n\t{f_dict}')
    elif not m_dict:
        # logging errors
        if debug: print(f'Log: The material for {f_name} ({mat}) is not in MAT_PROP_dict')
        #raise ValueError('Missing material dictionary data')
    
    temp_f_dict_list = []
    res = None

    if shape == 'Auto Select Shape':
        # Pool feature not implemented - therefore select one of the options for this
        stype = 'ASS'
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        as_dict = E2K_dict.get('AUTO SELECT SECTION LISTS',{}).get('AUTOSECTION',{}).get(f_name,{})
        if as_dict:
            f_dict2 = E2K_dict.get('FRAME SECTIONS',{}).get('FRAMESECTION',{}).get(f_name,{}).copy()
            if f_dict2:
                f_dict2['Note'] = ['Auto Select Shape']
                f_dict2['AUTOSELECTDESIGNTYPE'] = f_dict.get('AUTOSELECTDESIGNTYPE', '')
                f_dict2['POOL'] = as_dict.get('POOL_ID', 1)
                f_dict = f_dict2
    
    if shape == 'Auto Select List':
        stype = 'ASL'
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        pass
    elif shape == 'SD Section':
        stype = 'SDS'
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        pass  # this is addressed later
    elif 'Encasement' in shape:
        stype = 'EC'
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        enhance_encased_properties(f_dict, E2K_dict)
    elif shape == 'Nonprismatic':
        stype = 'NP'
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        temp_f_dict_list.append(f_dict)  # delme after TODO
    elif (('FILE' in f_dict.keys()) or  
            len(set(['ID', 'UNITS'] + list(f_dict.keys()))) < 5):   # a catalogue section
        stype = 'CAT'
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        res = enhance_CAT_properties(f_name, f_dict, m_dict,  
                        section_def_dict, sec_key_dict, prop_file, 
                        model_units, debug=debug)
    elif shape == 'NA':
        stype = 'NA'
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        enhance_CALC_properties(f_name, f_dict, m_dict, model_units, debug=debug)
    else:
        stype = 'CALC' # assume it is a standard section
        if debug: print(f'++ {f_name}: Mat: {mat}, Shape: {shape}, ShapeType: {stype}')
        enhance_CALC_properties(f_name, f_dict, m_dict, model_units, debug=debug)
        
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


def enhance_CALC_properties(f_name, f_dict, m_dict, 
                        model_units=Units('N', 'm', 'C'), debug=False):
    """
    
    """
    shape = f_dict.get('SHAPE')
    
    if shape in ('General', 'GENERAL', 'general'): 
        props = General_props_func(f_dict)
    elif shape in ('Steel I/Wide Flange', 'I/Wide Flange', 'WIDE FLANGE'): 
        props = I_props_func(f_dict)
    elif shape in ('Steel Pipe', 'Concrete Pipe', 'Pipe', 'PIPE'): 
        props = CHS_props_func(f_dict)
    elif shape in ('Filled Steel Pipe'): 
        props = CHS_props_func(f_dict)  # TODO temporary
    elif shape in ('Box/Tube','Steel Tube', 'Concrete Tube', 'Tube', 'TUBE'):
        props = RH_props_func(f_dict)
    elif shape in ('Filled Steel Tube'):
        props = RH_props_func(f_dict)  # TODO  temporary
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
    elif shape in ('Buckling Restrained Brace'): # BUCKLING RESTRAINED BRACE SECTIONS
        if debug: print(f'WARNING - check units of {f_name}: {f_dict}')  # TODO Fix units
        props = R_props_func(f_dict)  # TODO Temporary
    elif shape in ('NA'):
        props = {}
    else:
        # logging errors
        if debug: print(f'Log: No section ("shape") property enhancement functions found for {f_name}:\n', f_dict)
        props = {}
    
    # ('Filled Steel Tube') # 'FILLMATERIAL'
    #('Concrete Encasement Rectangle',): CER_props_func,
    #('Concrete Encasement Circle',): CEC_props_func,
    #('Nonprismatic',): NP_props_func, 
    #('SD Section',): SD_props_func,

    #if not (isinstance(props, dict) and props):
    #    print(f'shape: {shape}, f_dict: {f_dict}')
    for k, v in props.items():
        f_dict[k] = v
    
    # #### EDIT THIS #######
    sh_dict = convert_prop_units(f_dict, model_units.length)
    if sh_dict is None and debug:
        # logging errors
        print('Log (enhance_CALC_properties): convert_prop_units has failed with:')
        print(f'   shape_dict: {sh_dict}')
        print(f'   model_units: {model_units}')
        # pass
    # #### END EDIT ######
    
    area = f_dict.get('A')
    mat = f_dict.get('MATERIAL')
    if mat and m_dict:
        wt_density = m_dict.get('W') if m_dict.get('W') else m_dict.get('WEIGHTPERVOLUME')
        mat_type = m_dict.get('DESIGNTYPE') if (m_dict.get('W') and m_dict.get('DESIGNTYPE')) else m_dict.get('TYPE')
    else:
        wt_density = None
    # 5. Place section properties into the dictionary (with units)
    if wt_density and area:
        f_dict['Frame_Agg_Props'] = [Frame_Agg_Props(mat, mat_type.casefold(), wt_density, area)]


def enhance_CAT_properties(f_name, f_dict, m_dict,  
                        section_def_dict, sec_key_dict, prop_file, 
                        model_units=Units('N', 'm', 'C'), debug=False):
    """"""
    # For catalogue sections, the sections can be looked up.
    # but the units may need to be converted.
    # 1. Lookup correct file name using sec_key_dict lookup on lowercase name
    if prop_file is not None:
        file_base = splitext(basename(f_dict.get('FILE', prop_file)))[0]
        prop_file = sec_key_dict.get(file_base.casefold(), None)
    f_dict['FILE'] = prop_file # update 'FILE' with useful propfile
    # 2. Lookup section properties
    shape_dict = get_cat_sec_props(f_dict, section_def_dict)
    if shape_dict and isinstance(shape_dict, dict):
        # 3. Place section properties into the dictionary (with units)
        for k, v in shape_dict.items():
            f_dict[k] = v  # note that this will set 'UNITS' to the value in the catalog
        
        # 4. Gather material density and converted section area
        sh_dict = convert_prop_units(shape_dict, model_units.length)
        if sh_dict is None and debug:
            # logging errors
            print('Log (enhance_CAT_properties): convert_prop_units has failed with:')
            print(f'   shape_dict: {shape_dict}')
            print(f'   model_units: {model_units}')
            pass
        area = sh_dict.get('A')
        mat = f_dict.get('MATERIAL')
        if mat and m_dict:
            wt_density = m_dict.get('W') if m_dict.get('W') else m_dict.get('WEIGHTPERVOLUME')
            mat_type = m_dict.get('DESIGNTYPE') if (m_dict.get('W') and m_dict.get('DESIGNTYPE')) else m_dict.get('TYPE')
        else:
            wt_density = None
            if debug: 
                print(f'*** {f_name} - Weight density missing')
                print(f'    f_dict: {f_dict}')
                print(f'    m_dict: {m_dict}')
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
    if (s_dict.get('PROPTYPE','').casefold() == 'deck'): # DECK
        conc_mat = s_dict.get('CONCMATERIAL')
        mc_dict = MAT_PROP_dict.get(conc_mat, {})
        mc_type = get_mat_type(mc_dict)
        mc_unit_wt = get_weight_density(mc_dict)
        deck_mat = s_dict.get('DECKMATERIAL')
        md_dict = MAT_PROP_dict.get(deck_mat, {})
        md_type = get_mat_type(md_dict)
        md_unit_wt = get_weight_density(md_dict)
        
        additional_data = deck_props_func(s_dict)
        s_dict.update({k:v for k, v in additional_data.items() if k in ['P', 'D_AVE', 'T_AVE']})
        B = s_dict.get('DECKRIBSPACING')
        P = s_dict.get('P', 0)
        D_AVE = s_dict.get('D_AVE', 0)
        T_AVE = s_dict.get('T_AVE', 0)
        agg_props = []
        agg_props.append(Shell_Agg_Props(conc_mat, mc_type, mc_unit_wt, D_AVE))
        agg_props.append(Shell_Agg_Props(deck_mat, md_type, md_unit_wt, T_AVE))
        s_dict['Shell_Agg_Props'] = agg_props
        
    elif (s_dict.get('PROPTYPE','').casefold() == 'wall'): # WALL
        conc_mat = s_dict.get('MATERIAL')
        mc_dict = MAT_PROP_dict.get(conc_mat, {})
        mc_type = get_mat_type(mc_dict)
        mc_unit_wt = get_weight_density(mc_dict)
        wall_thickness = s_dict.get('WALLTHICKNESS')
        agg_props = []
        agg_props.append(Shell_Agg_Props(conc_mat, mc_type.casefold(), mc_unit_wt, wall_thickness))
        s_dict['Shell_Agg_Props'] = agg_props
        
    elif (s_dict.get('PROPTYPE','').casefold() == 'slab'): # SLAB
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


def FILE_PP(E2K_dict, debug=False):
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

        try:
            file_date = parser.parse(date_txt)
        except:
            # logging
            if debug: print('Log (File_PP): First date-parsing attempt failed')
            file_date = parser.parse(date_txt.split()[0])
        #print(file_path)
        #print(file_date)
        E2K_dict['FILEPATH'] = file_path
        E2K_dict['FILEDATE'] = file_date
        #return 0


def PROGRAM_PP(E2K_dict, debug=False):
    """Postprocesses E2K_dict to extract program title and version
    """
    prog_dict = E2K_dict.get('PROGRAM INFORMATION')
    if isinstance(prog_dict, dict):
        prog_info = prog_dict.get('PROGRAM')
        prog_title = list(prog_info.keys())[0]
        prog_ver = prog_info[prog_title].get('VERSION')
        if debug:
            print(f'(PROGRAM_PP) {prog_title}: {prog_ver}')
        E2K_dict['PROGRAM_TITLE'] = prog_title
        E2K_dict['PROGRAM_VERSION'] = prog_ver


def CONTROLS_PP(E2K_dict, debug=False):
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


def MAT_PROPERTIES_PP(E2K_dict, debug=False):
    """Post-process properties for materials - add numerical IDs.
    """
    MAT_PROP_dict = E2K_dict.get('MATERIAL PROPERTIES', {}).get('MATERIAL', {})
    dict_keys  = MAT_PROP_dict.keys()

    # define zero-weight material for dummy elements
    MAT_PROP_dict['ZERO_WT'] = {'TYPE': 'OTHER', 'WEIGHTPERVOLUME': 0, 'E':1, 'U': 0.3, 'ID': 1}
    
    for i, m_dict in enumerate(MAT_PROP_dict.values()):
        if m_dict.get('ID', 0) != 1:  # maintain the 'NONE' as number 1 
            m_dict['ID'] = i + 2  # because GSA does not include number zero
    
    if debug: [print(f'** {k}:  {v}') for k, v in MAT_PROP_dict.items()]


def FRAME_SECTIONS_PP(E2K_dict, section_def_dict, debug=False):
    """Post-process frame sections in E2K_dict so that properties are 
    available in that same dictionary.

    This is one of a set of post-processing functions

    Args:
        E2K_dict (dict): Dictionary of E2K data generated by parsing an ETABS
            text file (E2K, $ET)
        section_def_dict (dict): Dictionary of standard section properties
            generated by `build_section_dict()`

    Returns:
        (None)
    """
    
    prop_file_default = 'sections8' # for catalogue lookup
    temp_f_dict_list = [] # for logging examples
    sec_key_dict = {sec_name.casefold(): sec_name \
                    for sec_name in section_def_dict.keys()}
    prop_file = sec_key_dict[prop_file_default]
    
    if debug:
        print('(FRAME_SECTIONS_PP) Initially: ', E2K_dict.get('UNITS'), '...')
    model_units = E2K_dict.get('UNITS', Units('N', 'm', 'C'))
    if debug:
        print('... then model_units: ', model_units)
    
    FRAME_PROP_dict = E2K_dict.get('FRAME SECTIONS', {}).get('FRAMESECTION', {})
    
    # Define dummy frame section property for line members with property 'NONE'
    FRAME_PROP_dict['NONE'] = {'MATERIAL': 'ZERO_WT',  'SHAPE': 'NA', 'ID': 1, 'GWA': 'STD I(mm) 300 400 5 5'}

    for i, (f_name, f_dict) in enumerate(FRAME_PROP_dict.items()):
        if f_dict.get('ID', 0) != 1:  # maintain the 'NONE' as number 1 
            f_dict['ID'] = i + 2  # because GSA does not include number zero and `1` is for NONE
        #
        if debug and i < 3:
            print(f'{i} | {f_name} :  {f_dict}')

        # Add default units for GWA string generation
        # Note that this will be changed later (in enhance_CAT_properties)
        # if the catalog properties are based on a different system.
        f_dict['UNITS'] = E2K_dict.get('UNITS').length 
        
        if f_dict.get('FILE'): # check format and correct if necessary
            # Older files have filepath, so convert these to file base-name
            file_base = splitext(basename(f_dict.get('FILE', prop_file)))[0]
            prop_file = sec_key_dict.get(file_base.casefold(), None)
            f_dict['FILE'] = prop_file
        
        if debug and i < 3:
            print(f'{i} | {f_name} :  {f_dict}')
        
        if f_dict.get('A'):
            res = None  # properties have already been enhanced        
        else:
            res = enhance_frame_properties(f_name, f_dict, E2K_dict, 
            section_def_dict, sec_key_dict, prop_file, model_units, debug=debug)
        
        # update default prop_file (since this becomes the default)
        if f_dict.get('FILE'):
            prop_file = f_dict.get('FILE')
        #temp_f_dict_list.append(res) ## delme after TODO
    #print('\n', temp_f_dict_list) ## delme after TODO


def ENCASED_SECTIONS_PP(E2K_dict, debug=False):
    """Post-process properties for encased sections
    """
    MAT_PROP_dict = E2K_dict.get('MATERIAL PROPERTIES', {}).get('MATERIAL', {})
    
    FRAME_PROP_dict = E2K_dict.get('FRAME SECTIONS', {}).get('FRAMESECTION', {})
    
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
            embed_mat_mod = embed_m_dict.get('E') 
            
            encase_mat = f_dict.get('ENCASEMENTMATERIAL')
            encase_m_dict = MAT_PROP_dict.get(encase_mat, {})
            encase_unit_wt = encase_m_dict.get('W') \
                            if encase_m_dict.get('W') \
                            else encase_m_dict.get('WEIGHTPERVOLUME')
            encase_mat_type = encase_m_dict.get('DESIGNTYPE') if (encase_m_dict.get('W') and encase_m_dict.get('DESIGNTYPE')) else encase_m_dict.get('TYPE')
            encase_mat_mod = encase_m_dict.get('E') 
            eta_E = embed_mat_mod / encase_mat_mod
            eta_W = embed_unit_wt / encase_unit_wt
            
            if f_dict['SHAPE'].endswith('Rectangle'):
                encase_props = R_props_func(f_dict)
                if debug:
                    print(f'++ encase_props (R): {f_name} | {encase_props}')
            elif f_dict['SHAPE'].endswith('Circle'):
                encase_props = C_props_func(f_dict)
                if debug:
                    print(f'++ encase_props (C): {f_name} | {encase_props}')
            else:
                encase_props = {}
            
            
            agg_props = []
            if embed_props.get('UNITS'):
                model_units = E2K_dict['UNITS']
                embed_props = convert_prop_units(embed_props, model_units.length)
            
            embed_area = embed_props.get('A', None)

            if debug:                
                print(f'++ embed_props: {f_name} | {embed_props}' + ('** No Props! **' if embed_area is None else ''))
            
            if embed_area is None:
                embed_area = 0

            agg_props.append(Frame_Agg_Props(embed_mat, embed_mat_type.casefold(), embed_unit_wt, embed_area))
            
            encase_area = encase_props.get('A') - embed_area
            agg_props.append(Frame_Agg_Props(encase_mat, encase_mat_type.casefold(), encase_unit_wt, encase_area))
            
            f_dict['Frame_Agg_Props'] = agg_props
            
            for k, v in encase_props.items():
                f_dict[k] = v
            for prop, propmod in (('W', 'WMOD'), ('M', 'MMOD'), ):
                embed_prop = embed_props.get(prop)
                encase_prop = encase_props.get(prop)
                if embed_prop and encase_prop:
                    f_dict[propmod] = 1 + (eta_W - 1) * embed_prop / encase_prop
            for prop, propmod in (('A', 'AMOD'), ('I22', 'I2MOD'), ('I33', 'I3MOD'), ):
                embed_prop = embed_props.get(prop)
                encase_prop = encase_props.get(prop)
                if embed_prop and encase_prop:
                    f_dict[propmod] = 1 + (eta_E - 1) * embed_prop / encase_prop
            

def NONPRISMATIC_SECTIONS_PP(E2K_dict, debug=False):
    """Post-processing the properties of non-prismatic sections
    TODO: add something for this"""
    pass


def AUTO_SELECT_SECTION_LISTS_PP(E2K_dict, debug=False):
    """TODO: Combines lists of pool sections into the dictionary
      AUTOSECTION "ATB70C"  STARTSECTION "B70C-28SN"  
      AUTOSECTION "ATB70C"  "B70C-28SN"  "B70C-32SN"  "B70C-36SN"  "B70C-40SN"  "B70C-45SN"  "B70C-50SN"  
    -> {"ATB70C": {"STARTSECTION": "B70C-28SN", "POOL_ID": 1, "POOL": ["B70C-28SN", "B70C-32SN"], ... }, ...}
    NB provide POOL_ID (for lookup in GWA)
    """
    pass 


def BUCKLING_RESTRAINED_BRACE_SECTIONS_PP(E2K_dict, debug=False):
    pass # probably not needed - data is probably already structured


def CONCRETE_SECTIONS_PP(E2K_dict, debug=False):
    pass # probably not needed - data is probably already structured


def SD_SECTIONS_PP(E2K_dict, debug=False):
    """
    Post-process section designer data - non-standard sections
    
    Note that this should take place after processing the frame sections and will add data into the E2K_dict['FRAME SECTIONS'] dictionary
    Note also that if other SHAPETYPES are processed, 
    it will be necessary to do some subtraction...
    """
    
    MAT_PROP_dict = E2K_dict.get('MATERIAL PROPERTIES',{}).get('MATERIAL',{})
    
    SD_SECTION_dict = E2K_dict.get('SECTION DESIGNER SECTIONS',{}).get('SDSECTION',{})
    
    units = E2K_dict.get('UNITS')
    
    for sd_sect, sd_data in SD_SECTION_dict.items():
        shapes = sd_data.get('SHAPE')
        mat_area_list = []
        if shapes:
            for name, shape in shapes.items():
                # add other options for SHAPETYPES, such as custom embedded sections
                # add logic to cover when the SD Section contains multiple - need to combine GWA strings
                if shape.get('MATERIAL') and shape.get('SHAPETYPE') == 'POLYGON':
                    if isinstance(shape['X'], (list, tuple)) and isinstance(shape['Y'], (list, tuple)):
                        polyline = list(zip(shape['X'], shape['Y']))
                        poly_props = perim_full_props(polyline)
                        area = poly_props['A'] # perim_area_centroid(polyline)[0]
                    else:           # error-catching
                        area = 0
                    if area == 0 and debug:   # error-catching
                        err_msg = poly_props.get('** Error Message','') + f'\nSDSECTION: {sd_sect}, SHAPE: {name}'
                        print(err_msg)
                        continue
                        #raise ValueError(err_msg)
                    
                    perimeter = poly_props.get('P',0)
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
                    if isinstance(shape['X'], (list, tuple)) and isinstance(shape['Y'], (list, tuple)):
                        polyline = list(zip(shape['X'], shape['Y']))
                        poly_props = perim_full_props(polyline)
                        area = poly_props['A']
                    else:            # error-catching
                        area = 0
                    if area == 0 and debug:    # error-catching
                        err_msg = poly_props.get('Error Message','') + f'\nSDSECTION: {sd_sect}, SHAPE: {name}'
                        print(err_msg)
                        continue
                        #raise ValueError(err_msg)
                    
                    perimeter = poly_props.get('P', 0)
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


def SHELL_PROPERTIES_PP(E2K_dict, debug=False):
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
    
    # Define dummy frame section property for line members with property 'NONE'
    SHELL_PROP_dict['NONE'] = {'MATERIAL': 'ZERO_WT', 'ID': 1}


    # Enhance the properties with aggregated values
    for i, s_dict in enumerate(SHELL_PROP_dict.values()):
        if s_dict.get('ID', 0) != 1:  # maintain the 'NONE' as number 1 
            s_dict['ID'] = i + 2  # because GSA does not include number zero and `1` is for NONE
        enhance_shell_props(s_dict, MAT_PROP_dict)
    
    E2K_dict['SHELL PROPERTIES'] = {'SHELLPROP': SHELL_PROP_dict}


def STORIES_PP(E2K_dict, debug=False):
    """Postprocesses E2K_dict to add elevations to story data
    
    TODO: this will need to be revised to take 
    'Tower' into account
    """
    
    STORY_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('STORY',{})
    if STORY_dict:
        if E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('Story_Lists') is None:
            STORY_keys = STORY_dict.keys()
            E2K_dict['STORIES - IN SEQUENCE FROM TOP']['Story_Lists'] = dict()
            E2K_dict['STORIES - IN SEQUENCE FROM TOP']['Story_Lists']['Default'] = STORY_keys
                    
        Story_List_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('Story_Lists')
        
        for STORY_keys in Story_List_dict.values():    
            base = sum(STORY_dict[key].get('ELEV',0) for key in STORY_keys)
            heights = [STORY_dict[key].get('HEIGHT',0) for key in STORY_keys]
            relative_elevations = list(accumulate(heights[::-1]))[::-1]
            absolute_elevations = [base + relev for relev in relative_elevations]
            story_ID = 1 # len(STORY_keys)
            for key, relev, abs_elev in zip(STORY_keys, relative_elevations, absolute_elevations):
                STORY_dict[key]['ID'] = story_ID
                story_ID += 1
                STORY_dict[key]['RELEV'] = relev
                STORY_dict[key]['ABS_ELEV'] = abs_elev  



def POINTS_PP(E2K_dict, debug=False):
    """'POINT COORDINATES': Postprocesses E2K_dict to organise 
    points, coords into key, value pairs if they are not already.
    
    Dictionary Approach - note that coordinates are lumped 
    together in tuples of (X, Y, DeltaZ)
    """
    POINTS_dict = E2K_dict.get('POINT COORDINATES', {}).get('POINT', {})
    point_keys = list(POINTS_dict.keys()) #
    if isinstance(point_keys[0], (tuple, list)):
        POINTS_dict = {try_numeric(pt[0]): pt[1:] for pt, val in POINTS_dict.items() if val == dict()}
        # Need to reassign data back to E2K_dict
        E2K_dict['POINT COORDINATES']['POINT'] = POINTS_dict


def POINT_ASSIGNS_PP(E2K_dict, debug=False):
    """'POINT ASSIGNS': Postprocesses E2K_dict to add
    coordinates to every node.
    
    NOTE: not all points are present in this dictionary, so 
    additional values will be added as necessary by other 
    post-processing in the element assignations. Among other
    things, this means that they will miss out on being
    included in diaphragm constraint groups.
    """
    # Get reference to story elevations
    STORY_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('STORY', {})
    
    # Get reference to points coordinates
    POINTS_dict = E2K_dict.get('POINT COORDINATES', {}).get('POINT', {})
    
    # Get reference to diaphragms and set up GROUPS subdirectory
    DIAPHRAGMS_dict = E2K_dict.get('DIAPHRAGM NAMES', {}).get('DIAPHRAGM', {})
    
    # DIAPHRAGM_GROUPS_dict = {}
    if E2K_dict.get('DIAPHRAGM NAMES') is not None:
        if E2K_dict['DIAPHRAGM NAMES'].get('GROUPS', None) is None:
            E2K_dict['DIAPHRAGM NAMES']['GROUPS'] = {}
    DIAPHRAGM_GROUPS_dict = E2K_dict['DIAPHRAGM NAMES']['GROUPS']
    
           
    # I am not sure what userjoints are for... we could collect them here
    # if so, then the corresponding block ~30 lines down should be uncommented
    """DIAPHRAGM_USERJOINTS_dict = {}
    if E2K_dict.get('DIAPHRAGM NAMES', {}):
        if E2K_dict['DIAPHRAGM NAMES'].get('USERJOINTS', None) is not None:
            E2K_dict['DIAPHRAGM NAMES']['USERJOINTS'] = {}
            DIAPHRAGM_USERJOINTS_dict = E2K_dict['DIAPHRAGM NAMES']['USERJOINTS']"""

    # Get reference to Point Assignments
    NODES_dict = E2K_dict.get('POINT ASSIGNS', {}).get('POINTASSIGN', {})
    dict_keys  = NODES_dict.keys()
    
    # Check if dictionary has already been processed
    # If it has, we don't want this messing with the IDs
    if not NODES_dict[list(dict_keys)[0]].get('ID'):
        for i, (nd_key, nd_dict) in enumerate(NODES_dict.items()):
            point, story = nd_key
            coords = POINTS_dict[point]
            if len(coords) == 3:
                x, y, dz = coords
                nd_dict['DELTAZ'] = dz
            else:
                x, y = coords
                dz = 0
            #if STORY_dict[story].get('ABS_ELEV') is None:
            #    if debug: print(f'Story: {story}\n', STORY_dict)
            #if STORY_dict[story].get('ABS_ELEV') is None:
            #    if debug: print(f'Story: {story}\n', STORY_dict)
            abselev = STORY_dict[story]['ABS_ELEV']
            #abselev = POINTS_dict.get(point, None)
            
            nd_dict['COORDS'] = (x, y, abselev - dz)
            nd_dict['ID'] = i + 1

            # Add to Diaphragm groups
            if nd_dict.get('DIAPH', None) is not None: 
                diaph_key = (story, nd_dict.get('DIAPH'))
                # if key exists, extract values (a list), otherwise return empty list
                d_group = DIAPHRAGM_GROUPS_dict.get(diaph_key, [])
                # append current node to group list and assign to dictionary
                d_group.append(nd_key)
                DIAPHRAGM_GROUPS_dict[diaph_key] = d_group
    
            """# Add to Userjoints groups (not sure what these are used for...)
            # for now we can group them by story
            if nd_dict.get('USERJOINT', None) is not None: 
                uj_key = (story)
                # if key exists extract values (a list), otherwise return empty list
                uj_group = DIAPHRAGM_USERJOINTS_dict.get(uj_key, [])
                # append current node to group list and assign to dictionary
                uj_group.append(nd_key)
                DIAPHRAGM_USERJOINTS_dict[uj_key] = uj_group"""
    

def DIAPHRAGM_PP(E2K_dict, debug=False):
    """"""
    pass


def LINE_CONN_PP(E2K_dict, debug=False):
    """'LINE CONNECTIVITIES': Postprocesses E2K_dict to 
    extract element type and organise connection data.
    
    """
    LINES_dict = E2K_dict.get('LINE CONNECTIVITIES',{}).get('LINE',{})
    POINTS_dict = E2K_dict.get('POINT COORDINATES',{}).get('POINT',{})

    for k, v in LINES_dict.items():
        if v.get('Type') is None:
            pd_list = []
            for k2,v2 in v.items():
                # If there are multiple definitions of a LINE, provide a warning and use the last
                if isinstance(v2[0],(list, tuple)):
                    if debug:
                        print(f'WARNING: multiple points: {k}: {v}\n  v2 is {v2}')
                        [print(f'  N1: {vv[0]}: {POINTS_dict.get(vv[0])} | N2: {vv[0]}: {POINTS_dict.get(vv[1])}') for vv in v2]                    
                    v2 = v2[-1]
                pd_list.append(('Type', k2))
                pd_list.append(('N1', (v2[0],v2[2])))
                pd_list.append(('N2', (v2[1],0)))
            #print(pd_list)
            for k3, v3 in pd_list:
                LINES_dict[k][k3] = v3


def LINE_ASSIGNS_PP(E2K_dict, debug=False):
    """'LINE ASSIGNS': Postprocesses E2K_dict to add
    coordinates, lengths, areas, volumes and weights 
    to every line assignment.
    
    NOTE: not all points are present in the NODES dictionary, 
    so additional values will be added to NODES_dict as
    necessary.
    """
    my_log = []
    
    # Get reference to story elevations
    #main_key = 'STORIES - IN SEQUENCE FROM TOP'
    #sub_key = 'STORY'
    #STORY_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    STORY_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('STORY',{})
    Story_List_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('Story_Lists',{})
    
    
    # STORY_lookup - provide index, get story
    
    # story_flag = False
    #if STORY_dict:
    #    STORY_keys = STORY_dict.keys()
    #    STORY_lookup = {i+1: story for i, story in enumerate(list(STORY_keys)[::-1])}
        
    #    # STORY_reverse_lookup - provide story, get index
    #    STORY_reverse_lookup = {v:k for k, v in STORY_lookup.items()}
    #    story_flag = True

    
    # Get reference to points (only required if point is not referenced
    # in the POINT ASSIGN dictionary)
    POINTS_dict = E2K_dict.get('POINT COORDINATES', {}).get('POINT', {})
    
    # Check NODES_dict for adding new nodes to NODES_dict
    nodes_flag = False
    NODES_dict = E2K_dict.get('POINT ASSIGNS', {}).get('POINTASSIGN', {})
    if NODES_dict:
        node_max_orig = len(NODES_dict)
        next_node_id = len(NODES_dict) + 1  ###  NB ID is 1-based not 0-based  ###
        nodes_flag = True
    
    # Get reference to sections 
    FRAME_PROP_dict = E2K_dict.get('FRAME SECTIONS', {}).get('FRAMESECTION', {})
    if debug: 
        print('\nFRAME_PROP_dict: ')
        [print(f'*** {k}: {v}') for k, v in FRAME_PROP_dict.items()]
        print()
    
    # Check LINES_dict that is to be referenced
    LINES_dict = E2K_dict.get('LINE CONNECTIVITIES', {}).get('LINE', {})

    # Consolidate all the checks
    MEMBERS_dict = E2K_dict.get('LINE ASSIGNS').get('LINEASSIGN')
    #if MEMBERS_dict and story_flag and nodes_flag and LINES_dict:    
    if MEMBERS_dict and Story_List_dict and nodes_flag and LINES_dict:    
        # Lookup Node_1 & Node_2 and convert into NODE references
        for i, (key, mem_dict) in enumerate(MEMBERS_dict.items()):
            # Check if dictionary has already been processed
            # If it has, we don't want this messing with the IDs
            if mem_dict.get('ID') is not None:
                break
            if (i<3 or isinstance(mem_dict.get('SECTION'), list)) and debug: 
                print(f'i: {i} | key: {key}')
                print(mem_dict)
            line, story = key  # e.g. (B21, L32)
            
            # reference the Line definition to get generic connectivity
            line_dict = LINES_dict.get(line) 
            mem_dict['ID'] = i + 1    # set ID ###  NB ID is 1-based not 0-based  ###
            #story_index = STORY_reverse_lookup[story]
            line_pts = []
            mem_dict['MEMTYPE'] = line_dict.get('Type')
            
            # Process start and end nodes to reference N1 & N2 from the LINE CONNECTIVITY (the ETABS POINT defs)
            # and to generate N1 & N2, JT1 & JT2 for the individual line element (BEAM, COLUMN etc)
            #     NB For the member (memdict), N1, N2 are node ID integers (for use by GSA)
            #     and JT1, JT2 are the standard ETABS tuple, e.g. (185, 'L3')
            for n in ('1', '2'):
                point_n, drop_n = line_dict.get('N' + n)
                if drop_n == 0:    # both ends are on the same story level
                    story_n = story
                else:      # one end is on a different story level
                    #story_n = STORY_lookup.get(story_index - drop_n)
                    story_n = rel_story(story, Story_List_dict, drop_n, tag=str(key) + f' [{n}]', debug=debug)
                mem_dict['JT' + n] = (point_n, story_n)
                
                ndict = NODES_dict.get((point_n, story_n))
                if isinstance(ndict,dict):
                    mem_dict['N' + n] = int(ndict.get('ID'))
                    coords_n = ndict.get('COORDS')
                    mem_dict['COORDS' + n] = coords_n
                    line_pts.append(coords_n)
                else:
                    coords_rel = POINTS_dict.get(point_n, None)
                    sdict = STORY_dict.get(story_n, None)
                    if coords_rel and sdict:
                        deltaZ = 0 if (len(coords_rel) < 3) else coords_rel[2]
                        Z = sdict.get('ABS_ELEV') - deltaZ
                        coords_n = (coords_rel[0], coords_rel[1], Z)
                        NODES_dict[(point_n, story_n)] = {'ID': next_node_id, 'COORDS':coords_n}
                        mem_dict['N' + n] = next_node_id
                        mem_dict['COORDS' + n] = coords_n
                        line_pts.append(coords_n)
                        next_node_id += 1
                    else:
                        my_log.append(f'LINE ASSIGNS: Node lookup failed for {key} at N{n}: {(point_n, story_n)}')
            
            # add member length
            length = None
            clear_length = None
            if len(line_pts) == 2:
                length = dist3D(line_pts[0], line_pts[1])
                clear_length = length - mem_dict.get('LENGTHOFFI', 0) - mem_dict.get('LENGTHOFFJ', 0)
                mem_dict['L'] = length
                mem_dict['L_c'] = clear_length
            
            # add section area (needs access to section definition containing section areas etc
            S_data = mem_dict.get('SECTION')
            if isinstance(S_data,list) and debug: print('S_data:', S_data)
            f_dict = FRAME_PROP_dict.get(S_data,{})
            #f_dict = FRAME_PROP_dict.get(mem_dict.get('SECTION'),{})
            agg_props = f_dict.get('Frame_Agg_Props', [])
            agg_props2 = []
            
            propmod_w = mem_dict.get('PROPMODW', 1)
            if clear_length:
                for agg_prop in agg_props:
                    brb_dens = f_dict.get('BRBWEIGHT',0) / clear_length
                    wt_density = brb_dens if f_dict.get('SHAPE') == 'Buckling Restrained Brace' else agg_prop.wt_density
                    agg_props2.append(Agg_Props(
                        agg_prop.material, 
                        agg_prop.mat_type, 
                        wt_density,
                        clear_length, 
                        agg_prop.area, 
                        agg_prop.area * clear_length, 
                        agg_prop.area * clear_length * wt_density * propmod_w))
                mem_dict['Memb_Agg_Props'] = agg_props2
    
    
    if debug:    
        ## Debugging CHECKS ##
        node_max_new = len(NODES_dict)
        node_max_change = node_max_new - node_max_orig
        print(f'LINES: Number of nodes has changed from {node_max_orig} to {node_max_new}')
        print(f'    a change of {node_max_change}\n')

        print(f'Number of errors: {len(my_log)}')
        print(f'Number of members: {len(MEMBERS_dict)}\n')
        print(my_log)


def AREA_CONN_PP(E2K_dict, debug=False):
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


def AREA_ASSIGNS_PP(E2K_dict, debug=False):
    """'AREA ASSIGNS': Postprocesses E2K_dict to add
    coordinates, thicknesses, areas, volumes and weights 
    to every line assignment.
    
    NOTE: not all points are present in the NODES dictionary, 
    so additional values will be added to NODES_dict as
    necessary.
    """
    my_log = []
    

    ## == main processes == ##
    #main_key = 'AREA ASSIGNS'
    #sub_key = main_key[:-1].replace(' ','')
    SHELLS_dict = E2K_dict.get('AREA ASSIGNS',{}).get('AREAASSIGN',{})
    
    if not SHELLS_dict:
        return

    # Area Connectivities
    AREAS_dict = E2K_dict.get('AREA CONNECTIVITIES',{}).get('AREA',{})
    
    # Get reference to story elevations
    #main_key = 'STORIES - IN SEQUENCE FROM TOP'
    #sub_key = 'STORY'
    #STORY_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    STORY_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('STORY',{})
    Story_List_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('Story_Lists',{})
    
    # STORY_lookup - provide index, get story
    # story_flag = False
    # if STORY_dict:
    #    STORY_keys = STORY_dict.keys()
    #    STORY_lookup = {i+1: story for i, story in enumerate(list(STORY_keys)[::-1])}
    #    
    #    # STORY_reverse_lookup - provide story, get index
    #    STORY_reverse_lookup = {v:k for k, v in STORY_lookup.items()}
    #    story_flag = True
    
    # Get reference to points (only required if point is not referenced
    # in the POINT ASSIGN dictionary)
    #main_key = 'POINT COORDINATES'
    #sub_key = 'POINT'
    #POINTS_dict = get_E2K_subdict(E2K_dict, main_key, sub_key)
    POINTS_dict = E2K_dict.get('POINT COORDINATES', {}).get('POINT', {})
    
    # Check NODES_dict for adding new nodes to NODES_dict
    nodes_flag = False
    NODES_dict = E2K_dict.get('POINT ASSIGNS', {}).get('POINTASSIGN', {})
    if NODES_dict:
        node_max_orig = len(NODES_dict)
        next_node_id = len(NODES_dict) + 1  ###  NB ID is 1-based not 0-based  ###
        nodes_flag = True

    # Get reference to sections 
    #main_key = 'SHELL PROPERTIES'
    #sub_key = 'SHELLPROP'
    SHELL_PROP_dict = E2K_dict.get('SHELL PROPERTIES', {}).get('SHELLPROP', {})    

    all_OK = False # Final checks
    #if story_flag and nodes_flag and AREAS_dict:
    if Story_List_dict and nodes_flag and AREAS_dict:
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
            #story_index = STORY_reverse_lookup[story]
            area_pts = []
            Nn_list = [str(i+1) for i in range(num_pts)]
            SHELLS_dict[key]['NumPts'] = num_pts
            SHELLS_dict[key]['MEMTYPE'] = area_dict.get('Type')
            
            for n in Nn_list:
                point_n, drop_n = area_dict.get('N' + n)
                if drop_n == 0:
                    story_n = story
                else:
                    #story_n = STORY_lookup.get(story_index - drop_n)
                    story_n = rel_story(story, Story_List_dict, drop_n, tag=str(key) + f' [{n}]', debug=debug)
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
                    if agg_prop and agg_prop.thickness:   # error-catching
                        agg_props2.append(Agg_Props(
                            agg_prop.material, 
                            agg_prop.mat_type, 
                            agg_prop.wt_density, 
                            agg_prop.thickness, 
                            shell_area, 
                            agg_prop.thickness * shell_area if agg_prop.thickness else 0, 
                            agg_prop.thickness * shell_area * agg_prop.wt_density * propmod_w if (
                                agg_prop.thickness and agg_prop.wt_density) else 0))
                SHELLS_dict[key]['Memb_Agg_Props'] = agg_props2
        
        
    if debug:
        ## Debugging CHECKS ##
        node_max_new = len(NODES_dict)
        node_max_change = node_max_new - node_max_orig
        print(f'AREAS: Number of nodes has changed from {node_max_orig} to {node_max_new}')
        print(f'    a change of {node_max_change}\n')

        print(f'Number of errors: {len(my_log)}')
        print(f'Number of shells: {len(SHELLS_dict)}\n')
        print(my_log)


def LOAD_CASES_PP(E2K_dict, debug=False):
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


## ==================================
## ===  Geometry Post-processing  ===
## ==================================

def story_geometry(E2K_dict, find_loops = False, debug=False):
    """identify t-intersections for beams in a horizontal plane and 
    add intermediate nodes to the beam dictionary E2K_dict['LINE ASSIGNS']['LINEASSIGN']"""
    STORY_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('STORY',{})
    NODE_dict = E2K_dict.get('POINT ASSIGNS', {}).get('POINTASSIGN', {})
    LINE_dict = E2K_dict.get('LINE ASSIGNS', {}).get('LINEASSIGN',{})
    AREA_dict = E2K_dict.get('AREA ASSIGNS', {}).get('AREAASSIGN',{})
    if E2K_dict.get('DIAPHRAGM NAMES', None) is None:
        E2K_dict['DIAPHRAGM NAMES'] = {}
    if E2K_dict.get('DIAPHRAGM NAMES', {}).get('LOOPS') is None:
        E2K_dict['DIAPHRAGM NAMES']['LOOPS'] = {}
    DIAPHRAGM_LOOPS_dict = E2K_dict['DIAPHRAGM NAMES']['LOOPS']
    if E2K_dict.get('MODEL SUMMARY', None) is None:
        E2K_dict['MODEL SUMMARY'] = {}
    #print('E2K keys', E2K_dict.keys())
    MODEL_SUMMARY_dict = E2K_dict['MODEL SUMMARY']

    units = E2K_dict.get('UNITS')    
    length_tolerance = 0.001 * units_conversion_factor(('m', units.length)) # 1.0mm physical tolerance
        

    storylist = [''] + list(STORY_dict.keys())
    
    # Cycle through each story in the list of storeys to generate list of lines in the plane
    # TO DO: consider iterating over LINE_dict entries rather than storeys. 
    for upper_story, lower_story in zip(storylist[:-1], storylist[1:]): # = '5/F'
        if debug:
            print(f'upper storey: {upper_story}, lower storey: {lower_story}', end='|')
            
        # Extract joint pairs for 1D elements (beams, lines) in the lower storey from beam dictionary
        #   - stored as ELEM, JT1, JT2 - e.g. (('B1566', 'L076'), (3690, 'L076'), (7880, 'L076'))
        line_z_dict = {}  # organise lines by z-coordinate since a storey may have more than one layer of beams
        for k, v in LINE_dict.items():
            if (k[1] == lower_story and 
                    v['MEMTYPE'] in ('BEAM', 'LINE')):
                z1 = rounder(NODE_dict.get(v['JT1'])['COORDS'][2], length_tolerance)
                z2 = rounder(NODE_dict.get(v['JT2'])['COORDS'][2], length_tolerance)
                if z1 == z2:    # 1D element is horizontal
                    zd = line_z_dict.get(z1, [])
                    zd.append((k, v['JT1'], v['JT2']))
                    line_z_dict[z1] = zd

        # Extract joint lists for 2D elements (walls) in the lower storey from area dictionary
        #   - stored as ELEM, JT1, JT2, JT3 etc - e.g. (('W724', 'L076'), (7880, 'L076'), (8214, 'L076'), ...)
        for k, v in AREA_dict.items():
            if (k[1] in (upper_story, lower_story) and v['MEMTYPE'] in ('PANEL', 'AREA')):
                joints = [v['JT'+str(i+1)] for i in range(v.get('NumPts'))]
                z_list = [rounder(NODE_dict.get(jt)['COORDS'][2], length_tolerance) for jt in joints]
                for zkey in line_z_dict.keys():
                    h_edge_list = [k] + [jt for jt, z in zip(joints, z_list) if zkey == z]
                    if len(h_edge_list) == 3:
                        zd = line_z_dict.get(zkey, [])
                        zd.append(h_edge_list)
                        line_z_dict[zkey] = zd
                
        for line_list in line_z_dict.values():
            # NODE_Connected_Beams_dict = connectivity_dict_builder(line_list, True)
            
            # Dictionary for looking up connected nodes for any node 
            NODE_Connected_Nodes_dict = connectivity_dict_builder(line_list)
            
            # Create reduced lookup dictionary for node coordinates
            nc_dict = {n: NODE_dict.get(n)['COORDS'] for n in NODE_Connected_Nodes_dict}
            nID_dict = {n: NODE_dict.get(n)['ID']  for n in NODE_Connected_Nodes_dict}
            
            # Split lines where they are touched by other lines (T-intersection)
            if debug:
                print('split', end='|')
                if lower_story == 'L076':
                    print(r'******************')
                    print(line_list)
                    print()
            
            # length_tolerance = 0.001 * units_conversion_factor(('m', units.length)) # 1.0mm physical tolerance
            t_tol = 0.001

            new_line_list, split_beams_dict = split_T(line_list, nc_dict, t_tol=t_tol, len_tol=length_tolerance, ang_tol=0.001, debug=debug)
            
            # update LINE_dict by adding the intersecting nodes
            # this is for use by the GWA_writer
            # It is assumed that the intermediate nodes were sorted
            # by t-parameter
            for beam_ID, n_list in split_beams_dict.items():
                #beam_ID = (beam_name, lower_story)
                tnn_list = [(t, nd, nID_dict.get(nd, None)) for t, nd in n_list]
                bm_dict = LINE_dict.get(beam_ID)
                if bm_dict is not None:
                    #intersecting_beam_nodes_set = set([n for _, n in n_list])
                    #beam_node_set = b_dict.get('INTERMEDIATE_NODES', set())
                    #beam_node_set.update(intersecting_beam_nodes_set)
                    #b_dict['INTERMEDIATE_NODES'] = beam_node_set
                    #b_dict['INTERMEDIATE_IDS'] = [nID_dict.get(nd) for nd in beam_node_set if nID_dict.get(nd) is not None]
                    # add list of (t-parameter, node_name, node_number to 
                    intermediate_node_list = bm_dict.get('INTERMEDIATE_NODES', [])
                    sorted_list = sorted(set(intermediate_node_list + tnn_list), key=lambda x: x[0])                
                    
                    # eliminate duplicates - it is assumed that close nodes have already been eliminated
                    s_list = sorted_list[:1] + [t2 for t1, t2 in zip(sorted_list[:-1], sorted_list[1:]) if
                            (
                                (t1[1:] != t2[1:]) and   # eliminate dupicate nodes
                                ((t2[0]-t1[0]) > 0.1*t_tol)  # eliminate close nodes
                            )]

                    bm_dict['INTERMEDIATE_NODES'] = s_list
                    

            # rebuild the connected nodes dictionary to include the intersections
            if len(split_beams_dict) > 0:
                NODE_Connected_Nodes_dict = connectivity_dict_builder(new_line_list)
            
            # Find duplicate nodes and eliminate them
            #
            
            # Find overlapping beams and break them up and eliminate duplicates
            #
            
            # Find loops in each story
            if find_loops:
                length_factor = units_conversion_factor((units.length, 'm'))
                if debug:
                    print('loop', end = ' | ')        
                try:
                    loops_list = all_loops_finder(nc_dict, NODE_Connected_Nodes_dict, debug=debug)

                    # Adds loop definitions and areas to dictionary
                    diaph_loops = DIAPHRAGM_LOOPS_dict.get(lower_story, [])
                    story_summary_dict = MODEL_SUMMARY_dict.get(lower_story, {})
                    story_loop_area = story_summary_dict.get('Loop_Area_m2', 0)
                    new_area = 0

                    for loop in loops_list:
                        if loop not in diaph_loops:
                            diaph_loops.append(loop)
                            polyline = [nc_dict[node] for node in loop]
                            if debug: print('area', end = ' | ')
                            new_area += perim_area_centroid(polyline)[0]  * length_factor**2
                            if debug: print('write', end = ' | ')
                    
                    # updates loops to include new ones
                    DIAPHRAGM_LOOPS_dict[lower_story] = diaph_loops
                    
                    # Adds new loop areas to summary (refactored section)
                    story_summary_dict['Loop_Area_m2'] = story_loop_area + new_area
                    MODEL_SUMMARY_dict[lower_story] = story_summary_dict
                
                except:
                    if debug:
                        print('loop_failed', end = ' | ')        
                
                if debug:
                    print('end')        
            


def connectivity_dict_builder(edge_list, as_edges=False):
    """Builds connectivity dictionary for each vertex (node) - a list
    of connected nodes for each node.

    Args:
        edge_list (list): a list describing the connectivity
            e.g. [('E7', 'N3', 'N6'), ('E2', 'N9', 'N4'), ...]
        as_edges (bool): whether to return connected vertices / nodes or edges
    Returns:
        (dict): connectivity dictionary, each node is a key and the 
            value is a set of connected nodes
            e.g. {'N3': {'N6', 'N11', 'N7'}, 'N9': {'N4'}, etc}
    """
    connectivity_dict = {}
    for b, n1, n2 in edge_list:
        n_set = connectivity_dict.get(n1,set())
        n_set.add(b if as_edges else n2)
        connectivity_dict[n1] = n_set
        n_set = connectivity_dict.get(n2,set())
        n_set.add(b if as_edges else n1)
        connectivity_dict[n2] = n_set
    return connectivity_dict


def append_to_dict_list(dict_of_lists, key, item, unique=False, sort=False):
    """Utility to append items to a referenced list in a dictionary, and if the 
    keyed list is not present in the dictionary, it adds the key and an empty list
    and then adds the item to the list.
    """
    if dict_of_lists.get(key) is None:
        dict_of_lists[key] = []
    item_list = dict_of_lists.get(key, [])
    item_list.append(item)
    if unique:
        item_list = list(set(item_list))
    if sort:
        item_list = sorted(item_list)
    dict_of_lists[key] = item_list


def append_to_dict_set(dict_of_sets, key, item, sort=False):
    """Utility to append items to a referenced set in a dictionary, and if the 
    keyed set is not present in the dictionary, it adds the key and an empty set
    and then adds the item to the list.
    The same can be achieved by setting unique to True with `append_to_dict_list`
    """
    if dict_of_sets.get(key) is None:
        dict_of_sets[key] = set()
    item_set = dict_of_sets.get(key, set())
    item_set.add(item)
    item_list = list(set(item_set))
    if sort:
        item_set = set(sorted(item_list))
    dict_of_sets[key] = item_set


def split_T(line_ID_list, nc_dict, t_tol=0.001, len_tol=0.01, ang_tol =0.001, debug=False):
    """Returns a line_list modified to include new intersections.
    
    Calculates self intersections where one line touches another using a
    sweep algorithm along the x-axis.
    
    augmented_list is a list of lines defined as pairs of tuples - only the x & y
    coordinates are used in the processing, and additional data can be carried
    as demonstrated below:
        ((x1, y1), (x2, y2))
        ((x1, y1, z1), (x2, y2, z2))
        ((x1, y1), (x2, y2), text_string)

    Note that this will include duplicate items for the same beam ID

    Args:
        line_list (list):
        nc_dict (dict): 
        tol (float): tolerance for the t-parameters
        len_tol (float): tolerance for length
        ang_tol (float): angular tolerance
    Returns:
        tuple: 
            new_line_ID_list: original line_ID_list with expansion of
                lines that have intersections
                e.g. [('B5', 'N32', 'N16'), ('B8', 'N3', 'N7'), ('B8', 'N7', 'N9'), ]
            T_sorted_dict: dictionary of beams that have intersections
                returns a sorted list of t-parameters and node IDs
                e.g. {'B11': [(0.2, 'N3'), (0.4,'N13')], 'B21': [etc]}
    """
    augmented_list = []
    for b, n1, n2 in line_ID_list:
        n1_aug = tuple(list(nc_dict[n1]) + [n1])
        n2_aug = tuple(list(nc_dict[n2]) + [n2])
        augmented_list.append((n1_aug, n2_aug, b))
    
    my_ID = ('B435', 'R/F T2') # for debugging
    my_input = [x for x in augmented_list if x[-1] == my_ID]

    # sort the lines after flipping the nodes if node2_x < node1_x
    # - this adds an integer to indicate whether the line has been flipped
    sorted_list = sorted((pt1, pt2, *tail, 1) if pt1[0] < pt2[0] else (pt2, pt1, *tail, -1) 
            for pt1, pt2, *tail in augmented_list)
    
    line_stack = []
    T_dict = {}
    
    # Identify T-type intersections and generate list of nodes to insert
    # for each beam
    for line1 in sorted_list:
        #len1 = magNDx(line1, limit=3)
        x = line1[0][0]
        
        # line 1 debugging identifier
        _, _, *tail1, _ = line1
        tag1 = str(tail1[-1]) if len(tail1) > 0 else ''
        
        next_stack = []
        for line2 in line_stack:
            
            #len2 = magNDx(line2, limit=3)
            # line 2 debugging identifier
            _, _, *tail2, _ = line2
            tag2 = (' | ' + str(tail2[-1])) if len(tail2) > 0 else ''
        
            if line2[1][0] >= x:
                next_stack.append(line2)
                
                # Test for line intersections (crossings)
                # Note that intersection parameters (t1 & t2 t-parameters)
                # relate to the sorted lines

                crossing = line_intersection2D(line1, line2, True, length_tol=len_tol, ang_tol=ang_tol, debug=debug, tag=(tag1 + tag2))
                
                if crossing['type']  in ('enclosed', 'enclosing', 'overlapping',
                    ): # is this the right place to eliminate overlaps?
                    # === NOT USED ===
                    pass
                    # ================                    
                if crossing['type'] not in ('parallel', 'anti-parallel', 'separate', 'offset', 'anti-offset', 
                                            'error', 'other', 'enclosed', 'enclosing', 'overlapping',):
                    
                    # Debugging
                    flag = True if line1[-2] == my_ID or line2[-2] == my_ID else False
                    if flag and debug: 
                        print(my_ID, ':')
                        print(my_input)
                        print(f'line1: {line1}\nline2: {line2}\n{crossing}')
                    
                    # ==========================================================
                    # ===== Arrange line parameters for line orientation =======
                    # ==========================================================
                    # extract line parameters for intersection - 
                    # convert the parameters and nodes for the original beam orientation
                    if debug:
                        if (crossing.get('t1') is None) or (crossing.get('t2') is None):
                            print(f'Error in postprocessing \n  line1 = {line1} \n  line2 = {line2} \n {crossing}')
                    
                    t1 = crossing['t1'] if (line1[-1] == 1) else (1 - crossing['t1'])
                    t2 = crossing['t2'] if (line2[-1] == 1) else (1 - crossing['t2'])
                    
                    # these line parameters are for the unflipped definitions
                    #t1_0 = t1 if (line1[-1] == 1) else (1 - t1)
                    #t2_0 = t2 if (line2[-1] == 1) else (1 - t2)
                    
                    # beam parameters - some need to be flipped back
                    # beam 1 ID
                    b1_ID = line1[-2] 
                    # beam 1 node IDs
                    b1n1_ID = line1[0][-1] if (line1[-1] == 1) else line1[1][-1] 
                    b1n2_ID = line1[1][-1] if (line1[-1] == 1) else line1[0][-1]
                    # beam 2 ID
                    b2_ID = line2[-2]
                    # beam 2 node IDs
                    b2n1_ID = line2[0][-1] if (line2[-1] == 1) else line2[1][-1]
                    b2n2_ID = line2[1][-1] if (line2[-1] == 1) else line2[0][-1]
                    
                    # ====================================================
                    # ======   Identify which lines touch which   ========
                    # ====== and create lists of new connectivity ========
                    # ====================================================
                    if (t_tol < t2 < (1 - t_tol)):
                        if (-t_tol < t1 < t_tol):
                            # line1 end1 touches line2
                            append_to_dict_list(T_dict, b2_ID, (t2, b1n1_ID))
                        elif ((1 - t_tol) < t1 < (1 + t_tol)):
                            # line1 end2 touches line2
                            append_to_dict_list(T_dict, b2_ID, (t2, b1n2_ID))
                    elif (t_tol < t1 < (1 - t_tol)):
                        if (-t_tol < t2 < t_tol):
                            # line2 end1 touches line1
                            append_to_dict_list(T_dict, b1_ID, (t1, b2n1_ID))
                        elif ((1 - t_tol) < t2 < (1 + t_tol)):
                            # line2 end2 touches line1
                            append_to_dict_list(T_dict, b1_ID, (t1, b2n2_ID))
                    # ================================================
                    
        line_stack = next_stack.copy()
        line_stack.append(line1)
    
    # Create an updated list of beamIDs and nodeID_start and nodeID_end
    # Note that this will have multiple entries for each beamID where they
    # have been split for intersections, and they are sorted by t-parameter.
    # The sorting by `t` is important for creating the beam segments (GSA analysis layer) 
    T_sorted_dict = {k: sorted(v, key=lambda vv: vv[0]) for k, v in T_dict.items()}
    new_line_ID_list = []
    for line in line_ID_list:
        b, n1, n2 = line
        if len(T_dict.get(b,[])) == 0: # no change...
            new_line_ID_list.append(line)
        else: 
            n_list = [n1] + [n for _, n in T_sorted_dict[b]] + [n2]
            [new_line_ID_list.append((b, n_i, n_j)) for n_i, n_j in zip(n_list[:-1], n_list[1:])]
    
    return new_line_ID_list, T_sorted_dict


def beam_overlap(beam_1, beam_2, t_dict, tol = 0.0001):
    """Returns parametric coefficients for line2 ends relative to line_1
    line_1 & line_2 are tuples of tuples (2D), each with trailing IDs
    for beam and point. 

    t_dict must contain entries t21 & t22 which are the parametric
    locations of ends 1 and 2 of beam 2 relative to beam 1. Note that the 
    dictionary can be generated by `eng_utilities.line_overlap`.

    Args:
        tol (float): a relative tolerance for the parameter t
    """
    (*_, ptID_11), (*_, ptID_12), beamID_1 = beam_1
    (*_, ptID_21), (*_, ptID_22), beamID_2 = beam_2

    t21, t22 = [t_dict.get(k, None) for k in ('t21', 't22')]
    new_beam_list = []

    # If parameters are missing from the dictionary
    if (t21 is None) or (t22 is None): # no change
        return [(beamID_1, ptID_11, ptID_12), (beamID_2, ptID_21, ptID_22)]
    # If both ends of both beams are essentially the same, return beam 1 only
    elif (-tol < t21 < tol) and ((1 - tol) < t22 < (1 + tol)):
        return [(beamID_1, ptID_11, ptID_12)]
    # If both ends of both beams are essentially the same but anti-parallel, return beam 1 only
    elif (-tol < t22 < tol) and ((1 - tol) < t21 < (1 + tol)):
        return [(beamID_1, ptID_11, ptID_12)]
    # If start ends of both beams are close, and beams are parallel, t22 > 1
    elif (-tol < t21 < tol) and (t22 > 1):
        return [(beamID_2, ptID_11, ptID_12), 
                (beamID_1, ptID_12, ptID_22),]
    # If start ends of both beams are close, and beams are parallel, t22 < 1
    elif (-tol < t21 < tol) and (t22 < 1):
        return [(beamID_1, ptID_11, ptID_21), 
                (beamID_2, ptID_21, ptID_22),]
    # If start and end ends of beams 1 & 2 are close, and beams are anti-parallel, t12 > 1
    elif (-tol < t22 < tol) and (t21 > 1):
        return [(beamID_2, ptID_11, ptID_12), 
                (beamID_1, ptID_12, ptID_21),]
    # If start and end ends of beams 1 & 2 are close, and beams are anti-parallel, t12 < 1
    elif (-tol < t22 < tol) and (t21 < 1):
        return [(beamID_1, ptID_11, ptID_22), 
                (beamID_2, ptID_22, ptID_21),]
    # line 1 encloses line 2
    elif ((0 < t21 < 1) and (0 < t22 < 1)):
        if t21 >= t22:
            return [(beamID_1, ptID_11, ptID_21),
                    (beamID_2, ptID_21, ptID_22),
                    (beamID_1, ptID_22, ptID_12),]
        else:
            return [(beamID_1, ptID_11, ptID_22),
                    (beamID_2, ptID_22, ptID_21),
                    (beamID_1, ptID_21, ptID_12),]
    # line 2 encloses line 1
    elif ((t21 < 0 and t22 > 0) or (t22 < 0 and t21 < 0)):
        if t21 >= t22:
            return [(beamID_2, ptID_21, ptID_11),
                    (beamID_1, ptID_11, ptID_12),
                    (beamID_2, ptID_12, ptID_22),]
        else:
            return [(beamID_2, ptID_22, ptID_11),
                    (beamID_1, ptID_11, ptID_12),
                    (beamID_2, ptID_12, ptID_21),]
    # parallel overlap, beam_1 first
    elif ((0 < t21 < 1) and (t22 > 1)):
        return [(beamID_1, ptID_11, ptID_21),
                (beamID_1, ptID_21, ptID_12),
                (beamID_2, ptID_12, ptID_22),]
    # parallel overlap, beam_2 first
    elif ((t21 < 0) and (0 < t22 < 1)):
        return [(beamID_2, ptID_21, ptID_11),
                (beamID_2, ptID_11, ptID_22),
                (beamID_1, ptID_22, ptID_12),]
    # anti-parallel overlap, beam_1 first
    elif ((0 < t22 < 1) and (t21 > 1)):
        return [(beamID_1, ptID_11, ptID_22),
                (beamID_1, ptID_22, ptID_12),
                (beamID_2, ptID_12, ptID_21),]
    # anti-parallel overlap, beam_1 first
    elif ((t22 < 0) and (0 < t21 < 1)):
        return [(beamID_2, ptID_22, ptID_11),
                (beamID_2, ptID_11, ptID_21),
                (beamID_1, ptID_21, ptID_12),]
    # Separate - no change
    elif ((t21 > 1 and t22 > 1) or (t21 < 1 and t22 < 1)): 
        return [(beamID_1, ptID_11, ptID_12), (beamID_2, ptID_21, ptID_22)]
    else: # catchall - no change
        return [(beamID_1, ptID_11, ptID_12), (beamID_2, ptID_21, ptID_22)]


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
      for datum in data.get('Memb_Agg_Props', [])] 
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


def MEMBER_quantities_summary(E2K_dict, descending = True, debug=False):
    """Extracts the materials quantities and places them into 
    dictionaries inside the main dictionary with keys: 
        (story, member type, material type, material name)    
    """
    MEMBERS_dict = E2K_dict.get('LINE ASSIGNS', {}).get('LINEASSIGN', {})
    SHELLS_dict = E2K_dict.get('AREA ASSIGNS', {}).get('AREAASSIGN', {})
    if debug: print(f'{len(MEMBERS_dict):10,d}: 1D members \n{len(SHELLS_dict):10,d}: 2D shells')

    for name, M_dict in (('MEMBERS SUMMARY', MEMBERS_dict), ('SHELLS SUMMARY', SHELLS_dict)):
        sum_dict = quantities_summary(M_dict)
        STORY_dict = E2K_dict.get('STORIES - IN SEQUENCE FROM TOP', {}).get('STORY', {})
        if descending:
            story_order = STORY_dict.keys()
        else:
            story_order = list(STORY_dict.keys())[::-1]
        
        sum_dict_keys = sum_dict.keys()
        #
        story_ordered = [[k for k in sum_dict_keys if k[0] == s_key] for s_key in story_order]
        # 
        [one_list.sort(key = itemgetter(1, 2, 3)) for one_list in story_ordered]
        E2K_dict[name] = {k:sum_dict.get(k) for k in sum(story_ordered,[])}
        if debug: print(f'{len(E2K_dict[name]):10,d}: {name}')
        
        #d2list = []
        #[[d2list.append(k) for k in sum_dict.keys() if k[0] == k1] for k1 in story_order]
        #if descending:
            #E2K_dict[name] = {k:sum_dict.get(k) for k in d2list}
        #else:
        #    E2K_dict[name] = {k:sum_dict.get(k) for k in d2list[::-1]}


def STORY_quantities_summary(E2K_dict, dtype='Story', debug=False):
    """Extracts the materials quantities and places them into 
    dictionaries inside the main dictionary with keys: 
        (story, member type, material type, material name)    
    """
    data_types = ['Story', 'Eltype', 'Mattype', 'Mat']
    dnum = data_types.index(dtype)
    # ('15F', 'FLOOR', 'concrete', 'RC280'): {'length': 0.9, 'area': 210.8, 'volume': 37.944, 'weight': 91.17}
    # ('13F', 'BEAM', 'concrete', 'RC280'): {'length': 172.02, 'area': 5.22, 'volume': 39.6, 'weight': 95.38}
    # TODO calculate summaries in the Notebook...
    MEMBERS_qdict = E2K_dict.get('MEMBERS SUMMARY', {})
    SHELLS_qdict = E2K_dict.get('SHELLS SUMMARY', {})
    sum_dict = {}
    for k, v in MEMBERS_qdict.items():
        value = sum_dict.get(k[dnum],0)
        new_value = value + v.get('weight',0)
        sum_dict[k[dnum]] = new_value
        pass
    E2K_dict['STORY_WT_SUMMARY'] = sum_dict


def MODEL_quantities_summary(E2K_dict, descending = True, debug=False):
    """Extracts the materials quantities and places them into 
    dictionaries inside the main dictionary with keys: 
        (story, member type, material type, material name)    
    """
    # ('15F', 'FLOOR', 'concrete', 'RC280'): {'length': 0.9, 'area': 210.8, 'volume': 37.944, 'weight': 91.17}
    # ('13F', 'BEAM', 'concrete', 'RC280'): {'length': 172.02, 'area': 5.22, 'volume': 39.6, 'weight': 95.38}
    # TODO calculate summaries in the Notebook...
    MEMBERS_qdict = E2K_dict.get('MEMBERS SUMMARY', {})
    SHELLS_qdict = E2K_dict.get('SHELLS SUMMARY', {})
    """sum_dict = {}
    for k, v in MEMBERS_qdict.items():
        
        pass"""
    pass

