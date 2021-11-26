"""Plotting Utilities

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
