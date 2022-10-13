"""Code for extracting geometry data from ETABS text files (*.E2K & *.$ET).

Functions exist for pushing the data out to a GSA text file.

TODO:
- openings in floor diaphragms
- revise default parsing so that two keys at the beginning become nested keys
    of nested dictionaries. Also, single key and value become a dictionary
    but that when parsing a subsequent line, a check is carried out for a
    dictionary with a single value as well as looking for sub-keys.
- calculate floor area for frame without slabs (99%) (add strategy for choice)
- create a list of floor slabs that have the `DIAPH` parameter.
- set up analyses and combinations (50%)
- coordinate systems (25%)
- grid layouts (0%)
- functions (response spectra, ground motions etc) (0%)
- wind loads (0%)
- seismic loads (0%)
- logs (0%)
- add filled steel tubes / pipes (MATERIAL & FILLMATERIAL props & quantities) (10%)
- add section pools (20%)
- add buckling restrained beams (10%)
- embedded sections (80%)
- check deck properties
- add logging: At the moment there is no log kept of elements that do not "make sense".
  This could be useful for identifying how complete the record is.
- LINECURVEDATA - just found it and I didn't know what it was...
- sort out situation where embed is an SD section

DONE:
- split beams at intersections (100%)
- add point and area loads (including point loads on beams) (100%)
- NONE sections -> dummy

"""

#from os import listdir
from os.path import exists, isfile, join, basename, splitext
import pickle

from eng_utilities.general_utilities import is_numeric, try_numeric
from eng_utilities.geometry_utilities import *
from eng_utilities.section_utilities import build_section_dict
from eng_utilities.E2K_postprocessing import *
from collections import namedtuple

LoadKey = namedtuple('LoadKey', 'MEMBER STORY LOADCASE')


def is_key(string):
    """Checking whether element functions as a 'key' in the file
    
    >>> [is_key(x) for x in ['RELEASE', '"PINNED"', 'CARDINALPT', '8', 8, 8.8, '"32"']]
    [True, False, True, False, False, False, False]
    """
    return (not(is_numeric(string)) and (string.find('"') == -1))


def line_split(line):
    """Fastest line-splitter (should be fast for IronPython)"""
    line_list = []
    [line_list.append('"' + chunk + '"') if i%2==1 else line_list.extend(chunk.split()) for i, chunk in enumerate(line.split('"'))]
    return line_list


def gather(data):
    """Does a reverse generation of tuples for an E2K line
    """
    # This could be the location to sort out the POINT data format
    data_list = []
    temp_list = []
    for datum in data[::-1]:
        if is_key(datum):
            if len(temp_list) < 1:
                out = None
            elif len(temp_list) == 1:
                out = temp_list[0]
            else:
                out = tuple(temp_list)
            data_list.append((datum, out))
            temp_list = []
        else:
            #print(try_numeric(datum.replace('"','')))
            temp_list = [try_numeric(datum.replace('"',''))] + temp_list
    return data_list[::-1]


# If top-level dict matches first item and sub-dict doesn't, then 
def try_branch(a_dict, data_coll, debug=False, keyword=''):
    """When provided with a bunched line, it sets up  
    dictionaries and subdictionaries in a tree structure
    based on branching.
    If this is not appropriate, it merges the entries into
    a single dictionary"""
    # pick first pair
    a, b = data_coll[0]
    try_1 = a_dict.get(a)
    if try_1 is None:  # if there isn't an entry already
        #print(f'{a} Not found, add {a}:, {b}: & {data_coll[1:]}')
        a_dict[a] = {b:{k:v for k, v in data_coll[1:]}}
    
    elif a_dict[a].get(b) is not None:  #  try_merge
        #print(f'{a}  found')
        #print(f'OK : {a_dict[a]}  {b}  -> {a_dict[a].get(b)} therefore, try_merge')
        b_dict = a_dict[a][b]
        if b_dict == dict(data_coll[1:]): # ignore repeated lines
            pass
        else:
            try_merge(b_dict, data_coll[1:], debug=debug, keyword=keyword)
            a_dict[a][b] = b_dict.copy()
    else:  # try_branch (tested)
        #print(f'{a}  found')
        #print(f'Not: {a_dict[a]}  {b}  -> {a_dict[a].get(b)}')
        a_dict[a][b] = {k:v for k, v in data_coll[1:]}


def try_merge(a_dict, data_coll, debug=False, keyword=''):
    """When the line of data has a key that matches an existing one
    it merges the data into the dictionary under the existing key"""
    try:
        ## - Snip start
        if not isinstance(data_coll, (list, tuple)):
            if debug:
                print(f'In try_merge ({keyword}), data_coll is {data_coll} (type: {type(data_coll)})')
        elif not isinstance(data_coll[0], (list, tuple)):
            if debug:
                print(f'In try_merge ({keyword}), data_coll[0] is {data_coll} (type: {type(data_coll[0])})')
        elif data_coll[0][0] == 'SHAPE':
            # c_dict = a_dict.copy()
            try_branch(a_dict, data_coll, debug=debug, keyword=keyword)
            return
        ## - Snip end
    except:
        print(f'WARNING: ** In try_merge ({keyword}), data_coll is {data_coll} (type: {type(data_coll)})')
        print('WARNING: (cont\'d)) Possibly a case of "IndexError: tuple index out of range" **')

    for data in data_coll:
        try_1 = a_dict.get(data[0], None)
        if try_1 is not None:
            # ---
            if isinstance(try_1, list):
                try_1.append(data[1])
            else:
                try_1 = [try_1] + [data[1]]
            a_dict[data[0]] = try_1
            # ---
            
            # --- the following has been removed from the corresponding location just above
            """# if try_1 == data[1]: # data is two levels deep
                #print('data is two levels deep')
                pass
            else:
                if isinstance(try_1, list):
                    try_1.append(data[1])
                else:
                    try_1 = [try_1] + [data[1]]
                a_dict[data[0]] = try_1"""
            # ---

        else:
            a_dict[data[0]] = data[1]


def load_func(the_dict, line, debug=False): # a_dict is 
    loadclass = line[0][0]
    member, story = line[0][1]
    line_dict = dict(line)
    key = tuple([member, story, line_dict.get('LC')])
    
    if the_dict.get(loadclass) is None:
        the_dict[loadclass] = dict()
        if debug:
            print(f'Starting to parse {loadclass}')
    a_dict = the_dict[loadclass]
    #print('a_dict', a_dict)
    a_dict[key] = a_dict.get(key, []) + list(load_parser(line_dict))

    
def load_parser(d):
    """
    For loadclass = 'POINTLOAD', 'LINELOAD' or 'AREALOAD'
      LINELOAD  "B2141"  "5F"  TYPE "POINTF"  DIR "GRAV"  LC "LL_0.5"  FVAL 15  RDIST 0.4
      LINELOAD  "B2109"  "6F"  TYPE "UNIFF"  DIR "GRAV"  LC "DL"  FVAL 0.7
      LINELOAD  "C7"  "GF"  TYPE "TEMP"  LC "THN"  T -10
      AREALOAD  "A1"  "MEZZ"  TYPE "TEMP"  LC "THP"  T 10'
      AREALOAD  "F1"  "G/F"  TYPE "UNIFF"  DIR "GRAV"  LC "LL"  FVAL 0.005'
      AREALOAD  "F34"  "L1"  TYPE "UNIFLOADSET"  "BOH"
    """
    ltype = d.get('TYPE')
    direction = d.get('DIR', None)
    ldict = {'TYPE': ltype, 'DIR': direction}
    if ltype == 'TRAPF': 
        load_values = [try_numeric(d.get(item)) for item in ('FSTART', 'FEND', 'RDSTART', 'RDEND')]
        load_data = (load_values[2], load_values[0]), (load_values[3], load_values[1])
        ave_load = 0.5 * (load_values[0] + load_values[1]) * (load_values[3] - load_values[2])
        ldict.update({'DATA': load_data, 'AVE_LOAD': ave_load})
    elif ltype == 'UNIFF':
        load_value = try_numeric(d.get('FVAL'))
        load_data = ((0, load_value), (1, load_value))
        ave_load = load_value
        ldict.update({'DATA': load_data, 'AVE_LOAD': ave_load})
    elif ltype == 'POINTF':  # This is for beams
        load_data = [try_numeric(d.get(item)) for item in ('FVAL', 'RDIST')][::-1]
        ldict.update({'DATA': tuple(load_data)})    
    elif ltype == 'FORCE':  # This is for nodes
        forces = ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')
        load_data = [try_numeric(d.get(item)) for item in forces]
        ldict.update({'DATA': tuple(load_data)})    
    elif ltype == 'TEMP':  # This is for lines and shells with uniform temperature load
        temp_data = try_numeric(d.get('T',0))
        ldict.update({'DATA': temp_data})    
    #return {key:[ldict]}
    return [ldict]


def combo_func(the_dict, line):  
    line_key = line[0][0]
    combo_name = line[0][1]
    data_type = line[1][0] # e.g. 'LOAD', 'SPEC', 'COMBO', 'LOADCASE', 'DESIGN'
    line_dict = dict(line[1:])
    
    # if the line key is not already in the dictionary, add it
    if not the_dict.get(line_key):
        the_dict[line_key] = dict() # the_dict['COMBO']
        #print(f'...adding {line_key} to COMBO_dict')
    
    # make the line key the current reference
    a_dict = the_dict[line_key] # a_dict is the_dict['COMBO']
    
    # if the combination is not already in the dictionary, add it
    if not a_dict.get(combo_name):
        a_dict[combo_name] = dict()
        #print(f'...adding {combo_name} to {line_key} to COMBO_dict')
    
    # make the combination name the current reference
    b_dict = a_dict[combo_name] # b_dict is the_dict['COMBO']['COMBO1']
    
    # if the combination name is not already in the dictionary, add it
    if data_type == 'TYPE': # add type to the combination dictionary
        b_dict['TYPE'] = line_dict['TYPE']
    elif data_type == 'DESIGN': # add type to the combination dictionary
        # b_dict['DESIGN'] = line_dict['DESIGN']
        b_dict.update({k:v for k,v in line_dict.items()})
    else: # add the different load cases with their load factor for each datatype
        #c_dict.get(data_type, []) + list(tuple([line_dict[data_type], line_dict['SF']]))
        if not b_dict.get(data_type): # if there is no datatype 'SPEC'
            b_dict[data_type] = [tuple([line_dict[data_type], line_dict['SF']])]
        else:  # add the new data to the existing
            the_list = b_dict[data_type]
            the_list.append(tuple([line_dict[data_type], line_dict['SF']]))
            b_dict[data_type] = the_list



def add_to_dict_list(the_dict, key, value):
    value_list = the_dict.get(key,[])
    value_list.append(value)
    the_dict[key] = value_list


def story_func(the_dict, line, debug=False):
    """
    One of the challenges is that if a Tower has been defined
    this needs to be carried over from any previous lines (it
    is only defined once for each line and that then applies to 
    all following ones)
    NB `current_tower` needs to be defined in the current namespace
    NB2: It seems that in some versions of ETABS, if only one 
        Tower is defined the Tower prefix is not used...
    """
    # Keep a list of stories
    if not the_dict.get('Story_Lists'):
        the_dict['Story_Lists'] = dict()
    
    line_key = line[0][0] # 'STORY'
    # Choosing to use string numbers as numbers
    story_basic_name = line[0][1] # str(line[0][1])
    story_type = line[1][0] # e.g. 'HEIGHT', 'ELEV'
    line_dict = dict(line)  # NB STORY is retained as a key-value pair
    if line_dict.get('TOWER') is None and the_dict.get('TOWERS') is None:
        story_name = story_basic_name
        add_to_dict_list(the_dict['Story_Lists'], 'Default', story_name)
    elif line_dict.get('TOWER') is not None and the_dict.get('TOWERS') is None:
        tower = line_dict['TOWER']
        the_dict['TOWERS'] = [tower]
        story_name = tower + '-' + story_basic_name
        add_to_dict_list(the_dict['Story_Lists'], tower, story_name)
    elif line_dict.get('TOWER') is None and the_dict.get('TOWERS') is not None:
        tower = the_dict['TOWERS'][-1]
        line_dict['TOWER'] = tower
        story_name = tower + '-' + story_basic_name
        add_to_dict_list(the_dict['Story_Lists'], tower, story_name)
    else:  # both are not None
        new_tower = line_dict['TOWER']
        #towers = the_dict['TOWERS']
        #towers.append(new_tower)
        #the_dict['TOWERS'] = towers
        add_to_dict_list(the_dict, 'TOWERS', new_tower)
        story_name = new_tower + '-' + story_basic_name
        add_to_dict_list(the_dict['Story_Lists'], new_tower, story_name)
    
    # if the line key is not already in the dictionary, add it
    if not the_dict.get(line_key):
        the_dict[line_key] = dict() # the_dict['STORY']
        #print(f'...adding {line_key} to COMBO_dict')
    
    # make the line key the current reference
    a_dict = the_dict[line_key] # a_dict is the_dict['STORY']
    
    # if the story is not already in the dictionary, add it
    if a_dict.get(story_name) is None:
        a_dict[story_name] = {**line_dict}
        if debug:
            print(f'...adding {story_name} to {line_key} to STORY_dict')
        
    else:   # update
        a_dict.update({k:v for k,v in line_dict.items()})
    
    #return a_dict


# ====================================
# ======  Main Parsing Function ======
# ====================================

def E2KtoDict(E2K_model_path, debug=False, **kwargs):
    """Parses E2K text files and returns a dictionary.
    
    kwargs can be used to pass information into the function
    At the moment it is only used for the `debug` flag
    
    Args:
        E2K_model_path (str): this is a string containing the path
            to the E2K or $ET text file generated by ETABS. The parsing
            can process any ETABS text file, but the output will have
            features specific to the format produced.
    
    Returns:
        (dict): a dictionary containing parsed data from the ETABS text file
            This data is compatible with JSON and may be stored in this format.
    """
    debug = kwargs.get('Debug', False) or debug
    
    if debug:
        print('\n===== Start parsing E2K text file ==========')
        print(E2K_model_path, '\n')
    
    E2K_dict = dict()
    # the_dict = E2K_dict
    E2K_dict['ParseLog'] = []
    ParseLog_list = E2K_dict['ParseLog']

    ignore_lines = False
    # , encoding='utf8'
    with open(E2K_model_path, 'r') as E2K_file:
        for line in E2K_file:
            if line.startswith(r'$ File'):
                E2K_dict['File'] = {'Header': line[7:].strip()}
                ignore_lines = True

            elif line.startswith(r'$ LOG'):
                ignore_lines = True # stops import of log lines
                key = line[2:].strip() # removes ` $`
                E2K_dict[key] = dict()
                # Point `the_dict` to the relevant entry in E2K_dict
                the_dict = E2K_dict[key] 
                # Identify which parsing function to use
                the_func = lambda x, y: try_branch(x, y, debug=debug, keyword=key)

            #elif line.startswith(r'$ POINT COORDINATES'):
            # print(f'Starting to process {line.strip()} *')
            #    ignore_lines = False
            #    key = line[2:].strip() # removes `$ `
            #    E2K_dict[key] = dict()
            #    the_dict = E2K_dict[key]
            #    the_func = point_parse
            
            elif line.startswith(r'$ STORIES - IN SEQUENCE FROM TOP'):
                if debug:
                    print(f'Starting to process {line.strip()} *')
                ignore_lines = False
                key = 'STORIES - IN SEQUENCE FROM TOP' # line[2:].strip() # removes `$ `
                E2K_dict[key] = dict()
                the_dict = E2K_dict[key]
                the_func = lambda x, y: story_func(x, y, debug=debug)
            
            elif (line.startswith(r'$ POINT OBJECT LOADS') or 
                    line.startswith(r'$ FRAME OBJECT LOADS') or 
                    line.startswith(r'$ LINE OBJECT LOADS') or 
                    line.startswith(r'$ SHELL OBJECT LOADS') or
                    line.startswith(r'$ AREA OBJECT LOADS')):
                if debug:
                    print(f'Starting to process {line.strip()} *')
                ignore_lines = False
                key = line[2:].strip() # removes `$ `
                E2K_dict[key] = dict()
                the_dict = E2K_dict[key]
                the_func = lambda x, y: load_func(x, y, debug=debug)
            
            elif line.startswith(r'$ LOAD COMBINATIONS'):
                if debug:
                    print(f'Starting to process {line.strip()} *')
                ignore_lines = False
                key = line[2:].strip()
                E2K_dict[key] = dict()
                the_dict = E2K_dict[key]
                the_func = combo_func
            
            # Default parsing set up
            elif line.startswith(r'$'):
                if debug:
                    print(f'Starting to process {line.strip()}')
                ignore_lines = False
                key = line[2:].strip()
                E2K_dict[key] = dict()
                the_dict = E2K_dict[key]
                the_func = lambda x, y: try_branch(x, y, debug=debug, keyword=key)

            elif line.strip() == '':
                # Ignore blank lines
                pass
            
            # General parsing of non-keyword lines
            else:
                if ignore_lines:  # Ignore lines if flag is set to False
                    pass
                else:            ### This is where all the parsing is done ###
                    # line_split - breaks lines based on double quotes and spaces
                    # gather - does a reverse generation of tuples for an E2K line
                    dc = tuple(gather(line_split(line))) # 
                    the_func(the_dict, dc)  # the active dictionary is modified
    
    # Review story data for consistency
    is_consistent, log_text = story_consistency_check(E2K_dict, debug=debug)
    ParseLog_list.append(log_text)

    if not is_consistent:
        print(log_text)

    if debug:
        print(f'\n** E2K_dict Summary (E2KtoDict) ****')
        for k,v in E2K_dict.items():
            print(k)
            if isinstance(v, dict):
                if len(v) < 6:
                    [print(f'{len(vv):7d}  : {kk}') for kk, vv in v.items()]
                else:
                    print(f'{len(v):7d}  : {k}')
            elif isinstance(v, (list, tuple)):
                print(v[:6])
            
        print('===== Finished parsing E2K file to E2K_dict ==========')
    return E2K_dict


def story_consistency_check(E2K_dict, debug=False):
    """Checks for consistency in story keys. If keys are inconsistent then 
    calls for story information will fail.

    The stories (storeys) dictionary contains the original data ('STORY') and a derived
    dictionary to handle the fact that newer models contain 'TOWER' information that
    is typically added to all story calls in the POINT and member data. 
    """
    if debug: print('Consistency check - checking that all story references in points and members are matched in the story dictionary')
    # Review storey data for consistency
    len_story_lists = len(E2K_dict['STORIES - IN SEQUENCE FROM TOP'].get('Story_Lists',{}))
    
    story_list = E2K_dict['STORIES - IN SEQUENCE FROM TOP'].get('STORY',{}).keys()
    # Create a list of 'de-towered' story names (return None if there is no hyphen)
    story_list_1 = [str(s).split('-', 1)[-1] if ('-' in str(s)) else [None] for s in story_list]
    point_story_set = set()
    [point_story_set.add(k[1]) for k in E2K_dict['POINT ASSIGNS'].get('POINTASSIGN',{}).keys()]
    if debug:
        print('story_list = ', list(story_list))
        print('point_story_set = ', point_story_set)
    
    at_least_one = any(s in story_list for s in point_story_set) # Of the points in the model, at least one storey key is in the list of storeys 
    every_one = all(s in story_list for s in point_story_set) # Of the points in the model, all storey keys are in the list of storeys 
    at_least_one_1 = any(s in story_list_1 for s in point_story_set) # Of the points in the model, at least one storey key is in the de-towered list of storeys
    every_one_1 = all(s in story_list_1 for s in point_story_set) # Of the points in the model, all storey keys are in the de-towered list of storeys
    
    if every_one:
        # No keys need replacing
        result = (True, 'No keys need replacing')
    elif (not at_least_one) and every_one_1 and (len_story_lists == 1):
        # All keys need replacing
        # Split tower name from individual story keys 
        new_story_dict = {k.split('-', 1)[1]:v for k, v in E2K_dict['STORIES - IN SEQUENCE FROM TOP'].get('STORY',{}).items()}
        #new_story_lists_dict = {k:[vv.split('-', 1)[1] for vv in v] for k, v in E2K_dict['STORIES - IN SEQUENCE FROM TOP'].get('Story_Lists',{}).items()}
        # Replace tower key with 'Default' and split tower name from individual story keys 
        new_story_lists_dict = {'Default':[vv.split('-', 1)[1] for vv in v] for k, v in E2K_dict['STORIES - IN SEQUENCE FROM TOP'].get('Story_Lists',{}).items()}
        E2K_dict['STORIES - IN SEQUENCE FROM TOP']['STORY'] = new_story_dict
        E2K_dict['STORIES - IN SEQUENCE FROM TOP']['Story_Lists'] = new_story_lists_dict
        result = (True, 'All keys have been replaced')
    elif (not at_least_one) and every_one_1:
        pass
    else:
        # There is a problem...
        result = (False, 'Inconsistent keys')

    return result



def E2KtoDict_test(text):
    """Parses E2K text and returns a dictionary"""
    E2K_dict = dict()
    # the_dict = E2K_dict

    ignore_lines = False
    for line in text.split('\n'):
        if line.startswith(r'$ File'):
            E2K_dict['File'] = line[7:].strip()
            ignore_lines = True

        elif line.startswith(r'$ LOG'):
            ignore_lines = True
            key = line[2:].strip()
            E2K_dict[key] = dict()
            the_dict = E2K_dict[key]
            the_func = try_branch

        #elif line.startswith(r'$ POINT COORDINATES'):
        #    ignore_lines = True
        #    key = line[2:].strip()
        #    E2K_dict[key] = dict()
        #    the_dict = E2K_dict[key]
        #    the_func = point_parse

        elif (line.startswith(r'$ POINT OBJECT LOADS') or 
                line.startswith(r'$ FRAME OBJECT LOADS') or 
                line.startswith(r'$ SHELL OBJECT LOADS')):
            print(f'Starting to process {line.strip()}')
            ignore_lines = False
            key = line[2:].strip()
            E2K_dict[key] = dict()
            the_dict = E2K_dict[key]
            the_func = load_func
            
        elif line.startswith(r'$'):
            ignore_lines = False
            key = line[2:].strip()
            E2K_dict[key] = dict()
            the_dict = E2K_dict[key]
            the_func = try_branch

        elif line.strip() == '':
            pass
        else:
            if ignore_lines:
                pass
            else:
                #parse_line(line)
                #print(r'** ' + line)
                #E2K_list.append(tuple(gather(line_split(line))))
                dc = tuple(gather(line_split(line)))
                the_func(the_dict, dc)
    
    return E2K_dict


## ================================
## ===  Combined E2K Processes  ===
## ================================

def key_printout(E2K_dict, min_length=6):
    for k,v in E2K_dict.items():
        print(k)
        if isinstance(v, dict):
            if len(v) < min_length:
                [print(f'{len(vv):7d}  : {kk}') for kk, vv in v.items()]
            else:
                print(f'{len(v):7d}  : {k}')
        elif isinstance(v, list):
            print(f'{len(v):7d}  : {k}')
        else:
            print(f'{k}  : {v}')


def process_E2K_dict(E2K_dict, find_loops=False, debug=False, aggregation=False):
    """Carries out all the post-processing of the parsed E2K file
    Most importantly, this adds quantities in a new dictionary"""
    if debug: print('\n===== Starting Post-processing ==========')
    
    FILE_PP(E2K_dict, debug=debug)
    if debug: print('___ FILE_PP complete _____________')
    PROGRAM_PP(E2K_dict, debug=debug)
    if debug: print('___ PROGRAM_PP complete _____________')
    CONTROLS_PP(E2K_dict, debug=debug)
    if debug: print('___ CONTROLS_PP complete _____________')
    STORIES_PP(E2K_dict, debug=debug)
    if debug: print('___ STORIES_PP complete _____________')
    MAT_PROPERTIES_PP(E2K_dict, debug=debug)
    if debug: print('___ MAT_PROPERTIES_PP complete _____________')
    section_def_dict = build_section_dict(debug=debug)
    if debug: print('___ section_def_dict complete _____________')
    FRAME_SECTIONS_PP(E2K_dict, section_def_dict, debug=debug)
    if debug: print('___ FRAME_SECTIONS_PP complete _____________')
    ENCASED_SECTIONS_PP(E2K_dict, debug=debug)
    if debug: print('___ ENCASED_SECTIONS_PP complete _____________')
    SD_SECTIONS_PP(E2K_dict, debug=debug)
    if debug: print('___ SD_SECTIONS_PP complete _____________')
    # NONPRISMATIC_SECTIONS_PP(E2K_dict) # TODO
    SHELL_PROPERTIES_PP(E2K_dict, debug=debug)
    if debug: print('___ SHELL_PROPERTIES_PP complete _____________')
    POINTS_PP(E2K_dict, debug=debug)
    if debug: print('___ POINTS_PP complete _____________')
    POINT_ASSIGNS_PP(E2K_dict, debug=debug)
    if debug: print('___ POINT_ASSIGNS_PP complete _____________')
    LINE_CONN_PP(E2K_dict, debug=debug)
    if debug: print('___ LINE_CONN_PP complete _____________')
    LINE_ASSIGNS_PP(E2K_dict, debug=debug)
    if debug: print('___ LINE_ASSIGNS_PP complete _____________')
    AREA_CONN_PP(E2K_dict, debug=debug)
    if debug: print('___ AREA_CONN_PP complete _____________')
    AREA_ASSIGNS_PP(E2K_dict, debug=debug)
    if debug: print('___ AREA_ASSIGNS_PP complete _____________')
    LOAD_CASES_PP(E2K_dict, debug=debug) # post processing STATIC LOADS or LOAD PATTERNS
    if debug: print('___ LOAD_CASES_PP complete _____________')
    #LINE_LOAD_PP(E2K_dict, debug=debug) # not needed
    if aggregation is True:
        if debug: print('*** Quantities aggregation') 
        MEMBER_quantities_summary(E2K_dict, debug=debug)
        if debug: print('___ MEMBER_quantities_summary complete _____________')
        STORY_quantities_summary(E2K_dict, 'Story', debug=debug)
        if debug: print('___ STORY_quantities_summary complete _____________')
    else:
        if debug: print('*** No quantities aggregation') 
        
    story_geometry(E2K_dict, find_loops=find_loops, debug=debug)
    if debug: print('___ story_geometry complete _____________')
    """try:
        story_geometry(E2K_dict)
    except:
        print('"story_geometry" failed')"""
    # LOAD COMBINATIONS   # TODO
    # ANALYSIS TASKS  # TODO
    if debug: print('\n===== Post-processing finished ==========\n')
    

def run_all(E2K_model_path, get_pickle=False, save_pickle=True, find_loops=False, debug=False, aggregation=False, **kwargs):
    """Runs all functions for parsing and post-processing an ETABS text file
    It returns a dictionary that is in the format of the text file.
    Since processing can be time-consuming, it pickles the output 
    and will preferentially unpickle if 'get_pickle' is True"""

    debug = kwargs.get('Debug', False) or debug
    
    pickle_path = splitext(E2K_model_path)[0] + '.pkl'
    if exists(pickle_path) and get_pickle == True:
        if debug:
            print('** Extracting E2K_dict from pickle file ***')
        E2K_dict = pickle.load(open(pickle_path, 'rb'))
    else:
        if debug: print('** Parsing E2K file... ***')
        E2K_dict = E2KtoDict(E2K_model_path, debug=debug, **kwargs)
        if save_pickle:
            if debug: print('\n** Pickling E2K_dict **')
            pickle.dump(E2K_dict, open(pickle_path, 'wb'))
            if debug: print('-- First pickle file ' + 
                (f'exists\n{pickle_path}\n' if exists(pickle_path) else f'does NOT exist\n{pickle_path}\n'))

    
    if debug: print(f'** `run_all` transitioning to `process_E2K_dict`, debug = {debug}')
    process_E2K_dict(E2K_dict, find_loops=find_loops, debug=debug, aggregation=aggregation)
    
    if save_pickle:
        pickle_path_2 = splitext(E2K_model_path)[0] + '_2.pkl'

        if debug: print(f'\nE2K_dict is a {type(E2K_dict)}')
        if debug: key_printout(E2K_dict)
        pickle.dump(E2K_dict, open(pickle_path_2, 'wb'))
        print('second pickle dump succeeded')
        
        """try:
            pickle.dump(E2K_dict, open(pickle_path_2, 'wb'))
            print('\n** Second pickle dump succeeded')
        except:
            print('\n** Second pickle dump failed')"""
    
        print('-- Second pickle file ' + 
            (f'exists\n{pickle_path_2}\n' if exists(pickle_path_2) else f'does NOT exist\n{pickle_path_2}\n'))

    if debug:
        print(f'\n** E2K_dict Final Summary (run_all) ****')
        # key_printout(E2K_dict)

    return E2K_dict



def main():
    """directory_listing = listdir(f'..\samples')
    e2k_list = [join(f'..\samples', fl) for fl in directory_listing if (fl.casefold().endswith('e2k') or fl.casefold().endswith('$et'))]
    [print(e2k) for e2k in e2k_list] ;# Read in the E2K text file
    #E2K_path = r'..\samples\SBVC_kipin_722_00.e2k'
    E2K_path = r'..\samples\Shaw_Nmm_2017_ULS.e2k'
    #E2K_path = r'..\samples\cw1_kNm_722_01.e2k'
    #E2K_dict = E2KtoDict(E2K_path)
    #E2K_dict = run_all(E2K_path, get_pickle=False, Debug=True)
    E2K_dict = run_all(E2K_path, get_pickle=False, Debug=False)
    #E2K_dict = run_all(E2K_path, get_pickle=True) 

    pkl2_file = E2K_path.replace(".e2k","_2.pkl").replace(".$et","_2.pkl").replace(".E2K","_2.pkl").replace(".$ET","_2.pkl")
    print(f'\n=== Use these lines to import the model data: ===')
    print(f'import pickle')
    print(f'E2K_dict = pickle.load(open({pkl2_file}, "rb")') """
    pass

if __name__ == "__main__":
    main()
