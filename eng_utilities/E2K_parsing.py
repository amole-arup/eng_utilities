""""""

from eng_utilities.general_utilities import is_numeric, try_numeric
from geometry_utilities import *
from section_utilities import build_section_dict
import pickle
#from os import listdir
from os.path import exists, isfile, join, basename, splitext
from general_utilities import is_numeric, try_numeric
from E2K_postprocessing import *
from collections import namedtuple

LoadKey = namedtuple('LoadKey', 'MEMBER STORY LOADCASE')


def is_key(string):
    """
    
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
def try_branch(a_dict, data_coll):
    """When provided with a bunched line, it sets up  
    dictionaries and subdictionaries in a tree structure
    based on branching.
    If this is not appropriate, it merges the entries into
    a single dictionary"""
    # pick first pair
    a, b = data_coll[0]
    try_1 = a_dict.get(a)
    if not try_1:  # if there isn't an entry already
        #print(f'{a} Not found, add {a}:, {b}: & {data_coll[1:]}')
        a_dict[a] = {b:{k:v for k, v in data_coll[1:]}}
    elif a_dict[a].get(b):  #  try_merge
        #print(f'{a}  found')
        #print(f'OK : {a_dict[a]}  {b}  -> {a_dict[a].get(b)} therefore, try_merge')
        b_dict = a_dict[a][b]
        try_merge(b_dict, data_coll[1:])
        a_dict[a][b] = b_dict.copy()
    else:  # try_branch (tested)
        #print(f'{a}  found')
        #print(f'Not: {a_dict[a]}  {b}  -> {a_dict[a].get(b)}')
        a_dict[a][b] = {k:v for k, v in data_coll[1:]}


def try_merge(a_dict, data_coll):
    """When the line of data has a key that matches an existing one
    it merges the data into the dictionary under the existing key"""
    ## - Snip start
    if data_coll[0][0] == 'SHAPE':
        # c_dict = a_dict.copy()
        try_branch(a_dict, data_coll)
        return
    ## - Snip end
    for data in data_coll:
        try_1 = a_dict.get(data[0])
        if try_1:
            if try_1 == data[1]: # data is two levels deep
                #print('data is two levels deep')
                pass
            else:
                if isinstance(try_1, list):
                    try_1.append(data[1])
                else:
                    try_1 = [try_1] + [data[1]]
                a_dict[data[0]] = try_1
        else:
            a_dict[data[0]] = data[1]


def load_func(the_dict, line): # a_dict is 
    line_dict = dict(line)
    loadclass = list(line_dict.keys())[0]
    member, story = line_dict.get(loadclass)
    key = tuple([member, story, line_dict.get('LC')])
    
    if not the_dict.get(loadclass):
        the_dict[loadclass] = dict()
        print(f'Starting to parse {loadclass}')
    a_dict = the_dict[loadclass]
    #print('a_dict', a_dict)
    a_dict[key] = a_dict.get(key, []) + list(load_parser(line_dict))

    
def load_parser(d):
    """
    For loadclass = 'POINTLOAD', 'LINELOAD' or 'AREALOAD'"""
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
    elif ltype == 'POINTF':
        load_data = [try_numeric(d.get(item)) for item in ('FVAL', 'RDIST')][::-1]
        ldict.update({'DATA': tuple(load_data)})    
    elif ltype == 'FORCE':
        forces = ('FX', 'FY', 'FZ', 'MX', 'MY', 'MZ')
        load_data = [try_numeric(d.get(item)) for item in forces]
        ldict.update({'DATA': tuple(load_data)})    
    #return {key:[ldict]}
    return [ldict]


## NOTE: `Force Line` refers to a specific polyline
## format that is intended to represent properties
## along a line as a 2D shape anchored on a baseline
## (0,0) to (1,0). The function `build_force_line` 
## creates a 

def build_force_line(polyline, tol=1E-6):
    """Returns an open polyline suitable for 
    combining load profiles on a beam (trapezoids 
    defined by coordinates on a (0,0) to (1,0) baseline).
    """
    form = [[x, y] for x, y in polyline]
    #fix start of list
    if form[0][0] > tol:
        if form[0][1] > tol:
            form = [[0, 0]] + [[form[0][0], 0]] + form
        else:
            form = [[0, 0]] + form
    # fix end of list
    if (1 - form[-1][0]) > tol:
        if form[-1][1] > tol:
            form = form + [[form[-1][0], 0]] + [[1,0]]
        else:
            form = form + [[1, 0]]
    return form


def tidy_force_line(form, tol = 1E-6):
    """This will eliminate unnecessary duplicates
    However, this could mess with the matching process and should only
    be applied once everything has been processed"""
    #print('form length is ', len(form))
    if len(form) > 2:
        formout = [form[0]]
        v0 = sub2D(form[1], form[0])
        for pt1, pt2, pt3 in zip(form[0:-2], form[1:-1], form[2:]):
            v1 = sub2D(pt2, pt1)
            v2 = sub2D(pt3, pt1)
            sim = cos_sim2D(v1, v2) if  (mag2D(v1) > 2 * tol) else cos_sim2D(v0, v2)
            #print(sim, ': ', pt1, pt2, pt3, sub2D(pt2, pt1), sub2D(pt3, pt1))
            if abs(sim) < (1 - tol):
                formout.append(pt2)
            v0 = v1 if (mag2D(v1) > 2 * tol) else v0
        return formout + form[-1:]
    else:
        return form


def interpolate_force_line(form, x, tol=1E-6):
    """Interpolates a new point in a form polyline
    Used by the `add_force_line` function"""
    form_out = [form[0]]
    for pt1, pt2 in zip(form[:-1], form[1:]):
        if (x - pt1[0] > 0.5 * tol and 
            pt2[0] - x > 0.5 * tol):
            y = pt1[1] + (x - pt1[0]) * (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            form_out.extend(2 * [[x, y]])
        form_out.append(pt2)
    return form_out


def add_force_line(*forms): # form1, form2
    """
    Input is in the form of a line of coordinates
    uniformly increasing along the x-axis
    """
    x_vals = sorted(set(x for x, _ in sum(forms,[]))) # form1 + form2
    print('x_vals', x_vals)
    new_forms = []
    for form in forms:
        for x in x_vals:
            form = interpolate_force_line(form, x)
        print('form', form)
        new_forms.append(form)
    xs = [x for x, y_ in new_forms[0]]
    ys = [[y for _, y in form] for form in new_forms]
    #sum_ys = sum(n for n in zip(ys))
    #print('xs: ', len(xs), ': ', xs)
    #print('forms(0): ', len(new_forms[0]), ': ', new_forms[0])
    #print('forms(1): ', len(new_forms[1]), ': ', new_forms[1])
    #print('ys: ', len(ys), ': ', list(zip(*ys)))
    #return [[p1[0], p1[1] + p2[1]] for p1, p2 in zip(form1, form2)]
    new_ys = [sum(x for x in y) for y in zip(*ys)]
    return [[x, y] for x, y in zip(xs, new_ys)]


# ====================================
# ======  Main Parsing Function ======
# ====================================

def E2KtoDict(E2K_model_path, **kwargs):
    """Parses E2K text files and returns a dictionary.
    
    kwargs can be used to pass information into the function
    At the moment it is only used for the `debug`flag
    
    Args:
        E2K_model_path (str): this is a string containing the path
            to the E2K or $ET text file generated by ETABS. The parsing
            can process any ETABS text file, but the output will have
            features specific to the format produced.
    
    Returns:
        (dict): a dictionary containing parsed data from the ETABS text file
            This data is compatible with JSON and may be stored in this format.
    """
    debug = kwargs.get('Debug', False)
    E2K_dict = dict()
    # the_dict = E2K_dict

    ignore_lines = False
    with open(E2K_model_path, 'r') as E2K_file:
        for line in E2K_file:
            if line.startswith(r'$ File'):
                E2K_dict['File'] = {'Header': line[7:].strip()}
                ignore_lines = True

            elif line.startswith(r'$ LOG'):
                ignore_lines = True # stops import of log lines
                key = line[2:].strip() # remove ` $`
                E2K_dict[key] = dict()
                # Point `the_dict` to the relevant entry in E2K_dict
                the_dict = E2K_dict[key] 
                # Identify which parsing function to use
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
            
            # Default parsing set up
            elif line.startswith(r'$'):
                ignore_lines = False
                key = line[2:].strip()
                E2K_dict[key] = dict()
                the_dict = E2K_dict[key]
                the_func = try_branch

            elif line.strip() == '':
                # Ignore blank lines
                pass
            
            # General parsing of non-keyword lines
            else:
                if ignore_lines:  # Ignore lines if flag is set to False
                    pass
                else:            ### This is where all the parsing is done ###
                    dc = tuple(gather(line_split(line)))
                    the_func(the_dict, dc)  # the active dictionary is modified
    
    if debug:
        print(E2K_dict.keys())
    return E2K_dict


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

def process_E2K_dict(E2K_dict):
    FILE_PP(E2K_dict)
    PROGRAM_PP(E2K_dict)
    CONTROLS_PP(E2K_dict)
    STORIES_PP(E2K_dict)
    MAT_PROPERTIES_PP(E2K_dict)
    section_def_dict = build_section_dict()
    FRAME_SECTIONS_PP(E2K_dict, section_def_dict)
    ENCASED_SECTIONS_PP(E2K_dict)
    SD_SECTIONS_PP(E2K_dict)
    # NONPRISMATIC_SECTIONS_PP(E2K_dict) # TODO
    SHELL_PROPERTIES_PP(E2K_dict)
    POINTS_PP(E2K_dict)
    POINT_ASSIGNS_PP(E2K_dict)
    LINE_CONN_PP(E2K_dict)
    LINE_ASSIGNS_PP(E2K_dict)
    AREA_CONN_PP(E2K_dict)
    AREA_ASSIGNS_PP(E2K_dict)
    MEMBER_quantities_summary(E2K_dict)
    # LOADS   # TODO
    # GROUPS  # TODO
    

def run_all(E2K_model_path):
    """"""
    pickle_path = splitext(E2K_model_path)[0] + '.pkl'
    
    if exists(pickle_path) and True:
        E2K_dict = pickle.load(open(pickle_path, 'rb'))
    else:
        E2K_dict = E2KtoDict(E2K_model_path)
        pickle.dump(E2K_dict, open(pickle_path, 'wb'))
    
    process_E2K_dict(E2K_dict)
    
    return E2K_dict
    

