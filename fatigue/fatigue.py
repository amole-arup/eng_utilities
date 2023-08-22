""""""

from warnings import warn
from math import pi, tan, sin, cos, log, log10
#from os import path
#from sys import path as syspath
#from sys import version as pyver
#syspath.append(r'C:\Users\andrew.mole\OneDrive - Arup\Source\EtabsTxtParse')


## =====================================================================
### Damage Functions ############################################

#@nb.njit()
# def ssum(x):
#     total = 0.0
#     for items in x:
#         total += items
#     return total


def EN_1993_stress_cut_off(delta_sigma_C, **kwargs):
    """Returns the unfactored stress below which no fatigue occurs (EN 1993-1-9).
    The following may be set using keyword arguments (defaults shown):
    gamma_Mf: 1.1, gamma_Ff: 1.15, N_C: 2.0E6, N_D: 5.0E6, N_L:1.0E8
    m_1: 3, m_2: 5
    """
    gamma_Mf = kwargs.get('gamma_Mf', 1.1)
    gamma_Ff = kwargs.get('gamma_Ff', 1.15)
    N_C = kwargs.get('N_C', 2000000)
    N_D = kwargs.get('N_D', 5000000)
    N_L = kwargs.get('N_L', 100000000)
    m_1 = kwargs.get('m_1', 3)
    m_2 = kwargs.get('m_2', 5)
    delta_sigma_D = 1.0 * (N_C / N_D)**(1/m_1) * delta_sigma_C
    delta_sigma_L = 1.0 * (N_D / N_L)**(1/m_2) * delta_sigma_D
    return delta_sigma_L / gamma_Ff / gamma_Mf


#@nb.njit
def delta_s(N):
    """Davenport distribution of cycles due to buffeting wind"""
    return max(0,(0.7 * (log10(N))**2.0 - 17.4 * log10(N) + 100)/100)


def EN_1993_N_R_direct(delta_sigma_R, delta_sigma_C, **kwargs):
    """This carries out the same functions as EN_1993_N_R_func, except that it returns
    the answer directly rather than a vectorised function. This is faster for simple cases, 
    but the vectorised solution is much better for large dataframes."""
    debug = kwargs.get('debug', False)
    cut_off = kwargs.get('cut_off', True)
    gamma_Mf = kwargs.get('gamma_Mf', 1.1)
    gamma_Ff = kwargs.get('gamma_Ff', 1.15)
    design_life = kwargs.get('design_life', 50.0)
    N_C = kwargs.get('N_C', 2000000)
    N_D = kwargs.get('N_D', 5000000)
    N_L = kwargs.get('N_L', 100000000)
    m_1 = kwargs.get('m_1', 3)
    m_2 = kwargs.get('m_2', 5)
    delta_sigma_D_factor = 1.0 * (N_C / N_D)**(1/m_1)
    delta_sigma_L_factor = 1.0 * (N_D / N_L)**(1/m_2)
    design_life_factor = design_life / 50.0
    delta_sigma_D = delta_sigma_D_factor * delta_sigma_C
    delta_sigma_L = delta_sigma_L_factor * delta_sigma_D
    factored_stress = abs(delta_sigma_R) * gamma_Ff
    if debug:
        print('cut-off is active' if cut_off else 'cut-off is NOT active')
        uf_stress = delta_sigma_L / gamma_Mf / gamma_Ff
        print(f'delta_sigma_L is {delta_sigma_L:8.4g} (i.e. unfactored stress of {uf_stress:8.4g})')
        print(f's_D_factor is {delta_sigma_D_factor:8.4g}')
        print(f's_L_factor is {delta_sigma_L_factor:8.4g}')
        print(f'life_factor is {design_life_factor:8.4g}')
        print('factored_stress:', factored_stress)
    inv_N_cap =  min((gamma_Mf * factored_stress/delta_sigma_D)**m_1, (gamma_Mf * factored_stress/delta_sigma_D)**m_2) / N_D
    result = 0.0 if ((1 / inv_N_cap) >  N_L) else inv_N_cap * design_life_factor
    if debug:
        print(f'inv_N_cap: {inv_N_cap:8.4g}')
        L_compare = 'GREATER' if ((1 / inv_N_cap) >  N_L) else 'LESS'
        print(f'N_capacity ({1 / inv_N_cap:2,.0f}) is {L_compare} than N_L cut-off ({N_L:2,.0f})')
        print('result:', result if cut_off else inv_N_cap * design_life_factor)    
    return result if cut_off else inv_N_cap * design_life_factor


def EN_1993_N_R_func(**kwargs):
    """Returns a function to calculate number of cycles to failure when given axial stress, based on the S-N relationship in BS EN 1993-1-9
    Shear fatigue can be returned by passing m_1=5 as a keyword argument.
    Note that stresses will be multiplied by the material partial safety factor - gamma_Mf and the force factor gamma_Ff
        gamma_Mf - material partial safety factor (default 1.1)
        gamma_Ff - force partial safety factor (default 1.15)
        design_life - default 50 years
        N_C - default 2x10^6
        N_D - default 5x10^6
        N_L - default 10^8 (but inactivated if cut_off is False)
        m_1 - default 3
        m_2 - default 5
        cut_off - default True
        """
    gamma_Mf = kwargs.get('gamma_Mf', 1.1)
    gamma_Ff = kwargs.get('gamma_Ff', 1.15)
    design_life = kwargs.get('design_life', 50.0)
    N_C = kwargs.get('N_C', 2000000)
    N_D = kwargs.get('N_D', 5000000)
    N_L = kwargs.get('N_L', 100000000)
    m_1 = kwargs.get('m_1', 3)
    m_2 = kwargs.get('m_2', 5)
    cut_off = kwargs.get('cut_off', False)
    debug = kwargs.get('debug', False)

    #delta_sigma_D = (N_C / N_D)**(1/m_1) * delta_sigma_C / gamma_Mf
    delta_sigma_D_factor = 1.0 * (N_C / N_D)**(1/m_1)
    delta_sigma_L_factor = 1.0 * (N_D / N_L)**(1/m_2)
    design_life_factor = design_life / 50.0
    
    if debug:
        print('cut-off is active' if cut_off else 'cut-off is NOT active')
        print('delta_sigma_D_factor:', delta_sigma_D_factor)
        print('delta_sigma_L_factor:', delta_sigma_L_factor)
        print('design_life_factor:', design_life_factor)
        print('N_L:', N_L)
    
    
    #@vectorize(nopython=True)
    #@nb.jit(nopython=True)
    def inv_func(delta_sigma_R, delta_sigma_C):
        """It returns the inverse of the number of cycles to failure when provided a uniform direct axial stress - based on EN 1993-1-9
        - delta_sigma_R - the uniform stress
        - delta_sigma_C - the stress corresponding to failure at 2x10^6 cycles
        Note that it includes a cut-off at N_L (typically equal to 10^8) and that partial safety factors are applied
        """
        delta_sigma_D = delta_sigma_D_factor * delta_sigma_C
        delta_sigma_L = delta_sigma_L_factor * delta_sigma_D
        factored_stress = abs(delta_sigma_R) * gamma_Ff
        inv_N_cap =  min((gamma_Mf * factored_stress/delta_sigma_D)**m_1, (gamma_Mf * factored_stress/delta_sigma_D)**m_2) / N_D
        return 0.0 if ((1 / inv_N_cap) >  N_L) else inv_N_cap * design_life_factor 
    
    #@nb.jit(nopython=True)
    def inv_func_no_cutoff(delta_sigma_R, delta_sigma_C):
        """It returns the inverse of the number of cycles to failure when provided a uniform direct axial stress - based on EN 1993-1-9
        - delta_sigma_R - the uniform stress
        - delta_sigma_C - the stress corresponding to failure at 2x10^6 cycles
        Note that it does NOT include a cut-off at N_L (typically equal to 10^8) and that partial safety factors are applied
        """
        delta_sigma_D = delta_sigma_D_factor * delta_sigma_C
        delta_sigma_L = delta_sigma_L_factor * delta_sigma_D
        factored_stress = abs(delta_sigma_R) * gamma_Ff
        inv_N_cap =  min((gamma_Mf * factored_stress/delta_sigma_D)**m_1, (gamma_Mf * factored_stress/delta_sigma_D)**m_2) / N_D
        return inv_N_cap * design_life_factor 
    
    if cut_off:
        return inv_func
    else:
        return inv_func_no_cutoff


def DNVGL_RP_C203_N_R_func(**kwargs):
    """
    m_1
    log_a_1
    m_2
    log_a_2
    cut_off: False"""
    m_1 = kwargs.get('m_1', 3)
    log_a_1 = kwargs.get('log_a_1', 12.48)
    m_2 = kwargs.get('m_2', 5)
    log_a_2 = kwargs.get('log_a_2', 16.13)
    cut_off = kwargs.get('cut_off', False)
    
    #@vectorize(nopython=True)
    #@nb.jit(nopython=True)
    def inv_func(stress, not_used):
        """It returns the inverse of the number of cycles to failure when provided a uniform direct axial stress 
        - based on DNVGL_RP_C203
        - stress - the uniform stress
        - not_used - this is present to fit in with the standard function format
        Note that it does not include a cut-off or partial safety factors
        """
        inv_N_cap =  min((stress**m_1/log_a_1), (stress**m_2/log_a_2))
        return inv_N_cap
    return inv_func
    

from typing import Tuple, List, Union

# tuple[tuple[float], tuple[float]]]
def n_list_gen(method:str = 'power', ndiv: int = 0, nmax: int = 0, n_list:List[float] = []
            ) -> Tuple[Tuple[float], Tuple[float]]:
    """Generates band locations and band width to cover a range (0 to 10^nmax) 
    on the number line.
    
    This is a method for discretising (1D meshing) a range of numbers for
    discrete calculations that can be used to characterise the complete range.
    It is particularly used to generate lists of cycles corresponding to the 
    full wind spectrum for Davenport wind recurrence.
    `n_list` is an optional list of numbers that avoids the need to generate the list
    
    Args:
        method: power, ave, or low 
            - note that `low` is incorrect unless used with high numbers of divisions
        ndiv: number of bands that the spectrum is split up into 
        nmax: this is the exponent defining the maximum number on the range (10^nmax)
        n_list: this should normally be an empty list, unless the user wishes to specify the 
    
    Returns:
        Tuple of the list of cycles defining the bands and the list of band widths
    """
    if len(n_list) > 0: # a user-defined method
        n_list.sort() # to make sure that the numbers are sorted
        n_diff_list = tuple([1] + [m - n for m, n in zip(n_list[1:], n_list[:-1])])
    elif method == 'low':  # this is the method hardwired in the new fatigue spreadsheets (which are wrong)
        warn('The "low" method is unconservative and should not be used except for comparison.')
        n_list = (1,20,90,160,280,500,900,1600,2800,5000,9000,16000,28000,50000,90000,160000,280000,500000,900000,
            1600000,2800000,5000000,9000000,16000000,28000000,50000000,90000000,160000000,280000000,500000000)
        n_diff_list = tuple([1] + [m - n for m, n in zip(n_list[1:], n_list[:-1])])
    elif method == 'ave': # this becomes very accurate once ndiv >= 600
        ndiv = 600 if ndiv == 0 else ndiv
        nmax = 9 if nmax == 0 else nmax
        n_list = tuple([10**(float(n) * nmax/ndiv) for n in range(ndiv)])
        n_diff_list = tuple([n * (10**(nmax/2/ndiv) - 10**(-nmax/2/ndiv)) for n in n_list])
    else: # 'power'  - this is the method hardwired in the old fatigue spreadsheets
        ndiv = 30 if ndiv == 0 else ndiv
        nmax = 9 if nmax == 0 else nmax
        n_list = tuple([10**(float(n) * nmax/ndiv) for n in range(ndiv)])
        n_diff_list = tuple([n * nmax/ndiv * log(10) for n in n_list])
    return n_list, n_diff_list


def damage_func_gen(inv_damage_capacity_func, delta_s_n_func, **kwargs):
    """Returns a damage calculation function based only on direct stress (no thickness modifier)
       when provided with damage parameters (sigma_50, delta_sigma_C)
    
    Note that thickness modifiers should be applied to the 'detail category' before feeding it to the function
    Valid Keywords:
        method - 'power', 'ave' or 'low' (default 'power') Note that the "low" method is wrong
            "low" is a false method present in some spreadsheets
            "ave" becomes very accurate once ndiv >= 600 (but will be slow)
            "power" is the method hardwired in the old fatigue spreadsheets and is quite efficient
        output - 'value', 'list' or 'lol' (default 'value')
        debug - printing debugging output to the console (default False)

    Args:
        inv_damage_capacity_func: 

    Returns:
        damage_func
    """
    
    method = kwargs.get('method', 'power')
    output = kwargs.get('output', 'value') # or 'list'
    debug = kwargs.get('debug', False)
    
    ndiv = kwargs.get('ndiv', 0)
    nmax = kwargs.get('nmax', 0)
            
    n_list, n_diff_list = n_list_gen(method, ndiv, nmax)

    # Generate stress for a given number of cycles
    # - a recurrence relationship 
    s_list = tuple([delta_s_n_func(n) for n in n_list])
    
    # for use inside the damage function
    #inv_damage_capacity_func = inv_N_cap_func(gamma_Mf = gamma_Mf, gamma_Ff = gamma_Ff, m_1 = m_1, m_2 = m_2, N_C = N_C, N_D = N_D, N_L = N_L, design_life = design_life, debug=debug)
    #inv_damage_capacity_func = EN_1993_N_R_func(kwargs)
    
    
    #@nb.vectorize(nopython=True)
    def d_func_value(sigma_50, delta_sigma_C):
        """Returns fatigue damage based on detail category and stress (EN1993-1-9) and 50year nominal stress (not a range)
        Note that thickness modifiers should already have been applied to the 'detail category' 
        """
        print(f'List length = {len(n_list)}')
        return sum([(n * inv_damage_capacity_func((s * sigma_50), delta_sigma_C) if (s > 0) else 0) for n, s in zip(n_diff_list, s_list)])
    
    
    #@nb.vectorize(nopython=True)
    def d_func_list(sigma_50, delta_sigma_C):
        """Returns fatigue damage based on detail category and stress (EN1993-1-9) and 50year nominal stress (not a range)
        Note that thickness modifiers should already have been applied to the 'detail category' 
        """
        return [(n * inv_damage_capacity_func((s * sigma_50), delta_sigma_C) if (s > 0) else 0) for n, s in zip(n_diff_list, s_list)]
    

    #@nb.vectorize(nopython=True)
    def d_func_lol(sigma_50, delta_sigma_C):
        """Returns fatigue damage based on detail category and stress (EN1993-1-9) and 50year nominal stress (not a range)
        Note that thickness modifiers should already have been applied to the 'detail category' 
        """        
        return [(s, n, (n * inv_damage_capacity_func((s * sigma_50), delta_sigma_C) if (s > 0) else 0)) for n, s in zip(n_diff_list, s_list)]
    

    #@nb.vectorize(nopython=True)
    #def d_func_df(sigma_50, delta_sigma_C):
    #    """Returns fatigue damage based on detail category and stress (EN1993-1-9) and 50year nominal stress (not a range)
    #    Note that thickness modifiers should already have been applied to the 'detail category' 
    #    """        
    #    list_of_lists = [(s, n, (n * inv_damage_capacity_func((s * sigma_50), delta_sigma_C) if (s > 0) else 0)) for n, s in zip(n_diff_list, s_list)]
    #    df_out = pd.DataFrame(list_of_lists, columns = ['s','n','n/N'])
    #    df_out['N'] = df_out['n'] / df_out['n/N']
    #    df_out['stress'] = sigma_50 * df_out['s']
    #    return df_out
    
    if output == 'list':
        return d_func_list
    elif output == 'lol':
        return d_func_lol
    #elif output == 'df':
    #    return d_func_df
    else: #  output == 'value':
        return d_func_value



## =====================================================================
## Stress Functions
"""
These functions calculate the stress for different locations round a section. Currently they include calculations for CHS (pipes), CHS with single internal plates and I-sections, but not others.

The damage calculations are set up to only consider direct stress, so restructuring of the code would be required to cover shear stress.
"""

#@nb.vectorize(nopython=True)
def pk_stress(P_kN, M2_kNm, M3_kNm, A_m2, Z22_m3):
    return abs(0.001 * P_kN / A_m2) + 0.001 * (M2_kNm**2.0 + M3_kNm**2.0)**0.5 / Z22_m3


#@nb.vectorize(nopython=True)
def CHS_stress_calc(phi, psi, P, A, M2, Z2, M3, Z3):
    """Z2, Z3 refer to elastic section moduli (note that ETABS uses S22, S33)
    This may be used for any section with a circular perimeter provided the 
    principal axes are aligned with the local axes for which Z2 and Z3 are calculated."""
    return 0.001 * (P/A - M2 * sin(phi) / Z2 - M3 * cos(phi) / Z3)


#@nb.vectorize(nopython=True)
def CHS_plate_stress_calc(phi, psi, P, A, M2, Z2, M3, Z3):
    """Z2, Z3 are min and max principal elastic section moduli and psi is the angle of the plate (i.e. the major axis) from the 2-axis
    This is for cases where the CHS has an embedded plate and can cope with arbitrary rotations.
    The moments are transformed to suit the principal axes
    """
    Mu = M2 * cos(psi) + M3 * sin(psi)
    Mv = -M2 * sin(psi) + M3 * cos(psi)
    warn('Full QA required')
    return 0.001 * (P/A - Mu * sin(phi-psi) / Z2 - Mv * cos(phi-psi) / Z3)


#@nb.vectorize(nopython=True)
def H_maj_stress_calc(phi, psi, P, A, M2, Z2, M3, Z3):
    """Z2, Z3 are elastic section moduli, only valid for n_phi = 1 or 2"""
    return 0.001 * (P/A - M2 * sin(phi) / Z2 - M3 * cos(phi) / Z3)


#@nb.vectorize(nopython=True)
def H_min_stress_calc(phi, psi, P, A, M2, Z2, M3, Z3):
    """Z2, Z3 are elastic section moduli, only valid for n_phi = 1 or 2"""
    return 0.001 * (P/A + M2 * cos(phi) / Z2 - M3 * sin(phi) / Z3)


#@nb.vectorize(nopython=True)
def H_stress_calc(phi, psi, P, A, M2, Z2, M3, Z3):
    """Z2, Z3 are elastic section moduli, only valid for n_phi = 8
    The locations returned are 0, corner, 90, corner, 180, corner, 270"""
    if (abs(sin(8.0 * phi)) > 0.004):
        raise ValueError('n_phi should be 8')
    
    if (abs(phi) < 0.125 * pi) or (abs(phi - 2 * pi) < 0.125 * pi):
        return 0.001 * (P/A + M2 / Z2)
    elif (phi >= 0.125 * pi) and (phi < 0.375 * pi):
        return 0.001 * (P/A + M2 / Z2 - M3 / Z3)
    elif (phi >= 0.375 * pi) and (phi < 0.625 * pi):
        return 0.001 * (P/A - M3 / Z3)
    elif (phi >= 0.625 * pi) and (phi < 0.875 * pi):
        return 0.001 * (P/A - M2 / Z2 - M3 / Z3)
    elif (phi >= 0.875 * pi) and (phi < 1.125 * pi):
        return 0.001 * (P/A - M2 / Z2)
    elif (phi >= 1.125 * pi) and (phi < 1.375 * pi):
        return 0.001 * (P/A - M2 / Z2 + M3 / Z3)
    elif (phi >= 1.375 * pi) and (phi < 1.625 * pi):
        return 0.001 * (P/A + M3 / Z3)
    else:  # 1.625 <= phi < 1.875
        return 0.001 * (P/A - M2 / Z2 + M3 / Z3)
