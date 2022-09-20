"""Pandas DataFrame Utilities

These are for providing dataframes for testing.

In order not to consider pandas and numpy as a requirement,
the import of numpy and the functions are held
within a try-except, and dummy functions or results are provided.
"""

# Plotting Utility
try:
    import pandas as pd  # this is a standard alias
    print(f'Pandas version is {pd.__version__}')
    pd_OK = True
    
    
except:
    print(f'Pandas not installed')
    pd_OK = False


try:
    import pandasgui
    from pandasgui import show as _pd_show
    from pandasgui import __version__ as pdgver
    print(f'PandasGUI version is {pdgver}')
    
    def pd_show(*args, **kwargs):
        _pd_show(*args, **kwargs)
    
except:
    print(f'PandasGUI is not installed')
    
    def pd_show(*args, **kwargs):
        print('PandasGUI is not installed')
        return None


def dict_of_dicts_to_df(a_dict):
    """For dictionaries of dictionaries
    for dictionaries of lists try `to_df2`"""
    if pd_OK:
        return pd.DataFrame.from_dict(a_dict, orient='index')
    else:
        print('`to_df` failed. Pandas is not installed')
        return None


def to_df(a_dict):
    return dict_of_dicts_to_df(a_dict)


def dict_of_lists_to_df(a_dict):
    """For dictionaries of lists"""
    if pd_OK:
        dict2 = {}
        for k, v in a_dict.items():
            for i, vv in enumerate(v):
                kk = tuple(list(k) + [i])
                dict2[kk] = vv
        return pd.DataFrame.from_dict(dict2, orient='index')
    else:
        print('`to_df2` failed. Pandas is not installed')
        return None


def to_df2(a_dict):
    return dict_of_lists_to_df(a_dict)


#def E2K_dict_refs(E2K_dict, debug = False):
def df_dict(E2K_dict, debug = False):

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

    #return 

    MP_df = to_df(MAT_PROPS_dict)
    FP_df = to_df(FRAME_PROPS_dict)
    SP_df = to_df(SHELL_PROPS_dict)
    SPR_df = to_df(SPRING_PROPS_dict)
    N_df = to_df(NODE_dict)
    L_df = to_df(LINE_dict)
    A_df = to_df(AREA_dict)
    LP_df = to_df(LOADCASE_dict)
    PL_df = to_df2(POINT_LOAD_dict)
    LL_df = to_df2(LINE_LOAD_dict)
    AL_df = to_df2(AREA_LOAD_dict)
    LC_df = to_df(LOAD_COMBO_dict)
    S_df = to_df(STORY_dict)
    SL_df = to_df(Story_List_dict)
    D_df = to_df(DIAPHRAGMS_dict), 
    DG_df = to_df(DIAPHRAGM_GROUPS_dict), 
    DL_df = to_df(DIAPHRAGM_LOOPS_dict)
    
    return {
        'MAT_PROPS' : MP_df,
        'FRAME_PROPS' : FP_df,
        'SHELL_PROPS' : SP_df,
        'SPRING_PROPS' : SPR_df,
        'NODE' : N_df,
        'LINE' : L_df,
        'AREA' : A_df,
        'LOADCASE' : LP_df,
        'POINT_LOAD' : PL_df,
        'LINE_LOAD' : LL_df,
        'AREA_LOAD' : AL_df,
        'LOAD_COMBO' : LC_df,
        'STORY' : S_df,
        'Story_Lists' : SL_df,
        'DIAPHRAGMS' : D_df,
        'DIAPHRAGM_GROUPS' : DG_df,
        'DIAPHRAGM_LOOPS' : DL_df,
    }
