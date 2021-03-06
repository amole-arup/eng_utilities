""""""

from collections import namedtuple
from math import log10, ceil, floor


def is_numeric(numstring):
    try:
        return isinstance(float(numstring),float)
    except:
        return False


def try_numeric(numstring):
    """Converts strings containing numbers into numbers, else returns the string
    
    This is designed to help with the processing of ETABS text files
    
    Args:
        numstring (str, int, float): 
    
    Returns: (str, int, float): input converted to number, if possible 
    
    >>> [try_numeric(x) for x in ['RELEASE', '"PINNED"', 'CARDINALPT', '8', '2.2', 8, 2.2, '"32"']]
    ['RELEASE', '"PINNED"', 'CARDINALPT', 8, 2.2, 8, 2.2, '"32"']
    """
    if isinstance(numstring, str):
        try:
            return int(numstring)
        except ValueError:
            try:
                return float(numstring)
            except ValueError:
                return numstring
    else:
        return numstring

# try_numeric_ss('38'), try_numeric_ss('34.3'), try_numeric_ss('3f5'), 
# try_numeric('38'), try_numeric('34.3'), try_numeric('3f5')


def dict_sample_list(a_dict, n_sample = 4):
    """Sample dictionary data using even spacing - returns a list for printing"""
    data_len = len(a_dict)
    header = ['Printing {} entries evenly sampled from {} combinations:\n'.format(n_sample, data_len)]
    return header + ['** {} ** \n{}'.format(k,repr(a_dict[k])) for k in list(a_dict.keys())[1:-1:data_len // n_sample]]


def dict_sample_print(a_dict, n_sample = 4):
    """Sample dictionary data using even spacing - print to standard output"""
    data_len = len(a_dict)
    print('Printing {} entries evenly sampled from {} combinations:\n'.format(n_sample, data_len))
    [print('** {} ** \n{}'.format(k,repr(a_dict[k]))) for k in list(a_dict.keys())[1:-1:data_len // n_sample]]


def dict_slice(dictionary,start=None,end=None,step=None):
    """
    Slice a dictionary based on the (possibly arbitrary) order in which the keys are delivered
    :param dictionary:
    :param start: 
    :param end: 
    :param step: 
    :returns:
    """
    return {k:dictionary.get(k) for k in list(dictionary.keys())[start:end:step]}


### Units
Units = namedtuple('Units', 'force length temperature')
g_acc = 9.80665 # m/s^2

# Standard Unit Definitions

units_tuple = ('m', 'cm', 'mm', 'in', 'ft', 'yd', 
    'N', 'kN', 'MN', 'GN', 'kip', 'lb', 'kipf', 'kipm', 'lbf', 'lbm',
    'C', 'K', 'F', 'R', 'g', 'kg', 'ton', 'tonne', 'tonf', )


units_conv_base_dict = {
    ('cm', 'm'): 0.01, ('mm', 'm'): 0.001, ('cm', 'mm'): 10.0, 
    ('in', 'm'): 0.0254, ('in', 'cm'): 2.54, ('in', 'mm'): 25.4, 
    ('ft', 'm'): 0.3048, ('ft', 'cm'): 30.48, ('ft', 'mm'): 304.8, ('ft', 'in'): 12.0, 
    ('yd', 'm'): 0.9144, ('yd', 'cm'): 91.44, ('yd', 'mm'): 914.4, ('yd', 'ft'): 3.0, ('yd', 'in'): 36.0, 
    ('N', 'GN'): 1e-09, ('N', 'MN'): 1e-06, ('N', 'kN'): 0.001, ('kN', 'MN'): 0.001, 
    ('kgf', 'GN'): g_acc * 1e-09, ('kgf', 'MN'): g_acc * 1e-06, ('kgf', 'kN'): g_acc * 0.001, ('kgf', 'N'): g_acc, 
    ('tonf', 'GN'): g_acc * 1e-06, ('tonf', 'MN'): g_acc * 1e-03, ('tonf', 'kN'): g_acc, ('tonf', 'N'): g_acc * 1000.0, 
    ('lb', 'GN'): 4.4482216e-09, ('lb', 'MN'): 4.4482216e-06, ('lb', 'kN'): 4.4482216e-03, ('lb', 'N'): 4.4482216, 
    ('lbf', 'GN'): 4.4482216e-09, ('lbf', 'MN'): 4.4482216e-06, ('lbf', 'kN'): 4.4482216e-03, ('lbf', 'N'): 4.4482216, 
    ('kip', 'MN'): 0.0044482216, ('kip', 'kN'): 4.4482216, ('kip', 'N'): 4448.2216, 
    ('kipf', 'MN'): 0.0044482216, ('kipf', 'kN'): 4.4482216, ('kipf', 'N'): 4448.2216, 
    ('lb', 'mt'): 0.45359237e-06, ('lb', 't'): 0.45359237e-03, ('lb', 'kg'): 0.45359237, 
    ('lbm', 'mt'): 0.45359237e-06, ('lbm', 't'): 0.45359237e-03, ('lbm', 'kg'): 0.45359237, 
    ('kip', 'mt'): 0.00045359237, ('kip', 't'): 0.45359237, ('kip', 'kg'): 453.59237, 
    ('kipm', 'mt'): 0.00045359237, ('kipm', 't'): 0.45359237, ('kipm', 'kg'): 453.59237, 
    ('kg', 'mt'): 1e-06, ('kg', 't'): 0.001, ('t', 'mt'): 0.001, ('mt', 'mt'): 1.0,  
}

# add reverse lookup
units_conv_base_dict.update({(k[1], k[0]): 1.0 / v for k, v in units_conv_base_dict.items()})

# add units self lookup
units_conv_base_dict.update({(k[0], k[0]): 1.0 for k, _ in units_conv_base_dict.items()})

units_conv_dict = {**units_conv_base_dict}

# Dictionary to return Standard Unit Definitions
# -- 
units_lookup_dict = {u.casefold():u for u in units_tuple}

def units_conversion_factor(from_to):
    """from_to is a tuple containing the starting unit and the target unit
    """
    try:
        from_unit, to_unit = [units_lookup_dict.get(u.casefold()) for u in from_to]
    except:
        err_msg = f'units_lookup_dict.get(u.casefold()) failed for one of {from_to}'
        print(err_msg)
    from_unit, to_unit = [units_lookup_dict.get(u.casefold()) for u in from_to]
    if from_unit == to_unit:
        return 1.0
    else:
        return units_conv_dict.get((from_unit, to_unit), 1.0)


def unit_validate(unit):
    """Returns standardised unit when provided
    with case-insensitive input
    >>> unit_validate('M')
    'm'
    >>> unit_validate('KN')
    'kN'
    """
    return units_lookup_dict.get(unit.casefold())


def ci_lookup(the_key, the_dict, default=None):
    """Case-insensitive lookup
    
    Note that this will not work if keys are not unique in lower case
    - it will match the last of the matching keys
    
    >>> the_dict1 = {'Me': 'Andrew', 'You': 'Boris', 'Her': 'Natasha'}
    >>> ci_lookup('heR', the_dict1)
    'Natasha'
    """
    key_dict = {sec_name.casefold(): sec_name for sec_name in the_dict.keys()}
    if the_key.casefold() in key_dict.keys():
        return the_dict.get(key_dict[the_key.casefold()])
    else:
        return default


def rounder(n, e):
    """returns a number `n` rounded to match the magnitude of `e`"""
    return round(n, -int(floor(log10(e))))

