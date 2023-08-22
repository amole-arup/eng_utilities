"""A library for carrying out calculations to ASCE41-17
"""

from math import log, pi, sin, cos, tan, acos, asin, atan, atan2, log, log10
from itertools import accumulate


#import pandas as pd
def printt(table, h1 = 20, h2 = 12):
    """
    Printing tables where first column is text and the remainder are floats
    h1 is the 
    """
    for line in table:
        print('{:20s}'.format(line[0]), end = ' ')
        [print('{:8.4f}'.format(item), end = ' ') for item in line[1:]]
    return None


def interp(x, x1, x2, y1, y2):
    if y1 is None or y2 is None:
        return None
    elif x1 == x2:
        return 0.5 * (y1 + y2)
    else:
        return y1 + (x - x1) / (x2 - x1) * (y2 - y1)


def list_interp(val, list_1, list_2):
    """
    Returns an interpolated value based on a list of reference values and a list of target values
    val is the lookup value
    list_1 is the lookup list
    list_2 is the result list with values corresponding to those on list_1
    """
    i_list = [i for i, n in enumerate(list_1) if n == val]
    if i_list:
        return list_2[i_list[0]]
    elif val <= list_1[0]:
        return list_2[0]
    elif val >= list_1[-1]:
        return list_2[-1]
    else:
        index_list = [i for i, (x1, x2) in enumerate(zip(list_1[:-1], list_1[1:])) if val >= x1 and val <= x2]
        if len(index_list) == 1:
            j = index_list[0]
            return interp(val, list_1[j], list_1[j+1], list_2[j], list_2[j+1])
        else:
            return None


def k_gen(T):
    """Returns the coefficient for the vertical distribution of lateral force (Eq 4-2a)"""
    return max(1.0, min(2.0, 0.5 * T + 0.75))


def storey_force_coeffs(elev_list, weight_list, k = 1.0):
    """
        Inputs:
            elev_list - list of storey elevations (m)
          weight_list - list of storey weights (kN)
    """
    divisor = sum(w * h**k for w, h in zip(weight_list, elev_list))
    return [w * h**k / divisor for w, h in zip(weight_list, elev_list)]


def storey_force_distribution(V, elev_list, weight_list, k = 1.0):
    """
        Inputs:
            V  - base shear in kN
            elev_list - list of storey elevations (m)
          weight_list - list of storey weights (kN)
    """
    divisor = sum(w * h**k for w, h in zip(weight_list, elev_list))
    return [w * h**k * V / divisor for w, h in zip(weight_list, elev_list)]


def storey_forces(force_list, rev = True):
    """Returns a list of accumulated storey forces - i.e. storey shears
    NB lists have to be reversed before applying the accumulate function"""
    shear_list_rev = accumulate(force_list[::-1] if rev else force_list)
    return list(shear_list_rev)[::-1] if rev else list(shear_list_rev)


def storey_shear_moment(force_list, height_list, rev = True):
    """Returns a tuple of lists containing storey shear and storey moment
    (from ground up, increasing z if 'rev' is True)"""
    shear_list_rev = list(accumulate(force_list[::-1] if rev else force_list))
    v_h_rev = [v * h for v, h in zip(shear_list_rev, height_list[::-1] if rev else height_list)]
    mom_list_rev = list(accumulate([0]+v_h_rev[:-1]))
    return shear_list_rev[::-1] if rev else shear_list_rev, mom_list_rev[::-1] if rev else mom_list_rev


def shear_coords(elevs, shears):
    """Returns a list of (z,V) tuples for plotting storey shear diagrams
    Inputs are lists of elevations and shear values
    NB. Interleaving is required because of the steps in the shear diagram"""
    l_1 = list(zip(elevs[1:],shears[1:]))
    l_2 = list(zip(elevs,shears[1:] + [0.0]))
    out = []
    while l_2:
        out.append(l_2.pop()) if l_2 else None
        out.append(l_1.pop()) if l_1 else None
    return out[::-1]


# -----------------------------------------------------------------
# ----- Seismic Hazard --------------------------------------------
# -----------------------------------------------------------------

C_t_dict = {'S1':0.035, 'S1a':0.035, 'C1':0.018, 'S2':0.030, 'S2a': 0.030}
C_t_get = lambda x: C_t_dict.get(x, 0.020)  # a lookup function that returns default of 0.020

beta_dict = {'S1':0.80, 'S1a':0.80, 'C1':0.90}
beta_get = lambda x: beta_dict.get(x, 0.75)  # a lookup function that returns default of 0.75

unit_l_dict = {'m': 1.0, 'cm': 0.01, 'mm':0.001, 'in':0.0254, 'ft':0.3048, 'yd':0.9144}
unit_l_get = lambda x: unit_l_dict.get(x, None)  # a lookup function that returns default of None

def T_code_get(s_type, height, units = 'm'):
    """Returns the code period in seconds for a given structure type and height Eqn 4-4"""
    C_t = C_t_get(s_type)
    beta = beta_get(s_type)
    u_fac = unit_l_get(units)
    ft_fac = unit_l_get('ft')
    return C_t * height ** beta if units == 'ft' else C_t * (u_fac / ft_fac * height) ** beta


B_1_func = lambda x: 4.0 / (5.6 - log(100 * x)) # for x as damping ratio (not percentage)

# F_a and F_v taken from NEHRP 2015
F_a_headers = (0.25,0.5,0.75,1.0,1.25,1.5)
F_v_headers = (0.1,0.2,0.3,0.4,0.5,0.6)
F_a_dict = {'A': (0.8,0.8,0.8,0.8,0.8,0.8),
           'B': (0.9,0.9,0.9,0.9,0.9,0.9),
           'C': (1.3,1.3,1.2,1.2,1.2,1.2),
           'D': (1.6,1.4,1.2,1.1,1.0,1.0),
           'E': (2.4,1.7,1.3,None,None,None),
           'F': (None,None,None,None,None,None)}
F_v_dict = {'A': (0.8,0.8,0.8,0.8,0.8,0.8),
           'B': (0.8,0.8,0.8,0.8,0.8,0.8),
           'C': (1.5,1.5,1.5,1.5,1.5,1.4),
           'D': (2.4,2.2,2.0,1.9,1.8,1.7),
           'E': (4.2,3.3,2.8,2.4,2.2,2.0),
           'F': (None,None,None,None,None,None)}


def site_coeffs(site_class, S_s, S_1):
    """Returns F_a, F_v when provided with site class (A-F) and S_s and S_1 values"""
    F_a_list = F_a_dict[site_class]
    F_v_list = F_v_dict[site_class]
    return list_interp(S_s, F_a_headers, F_a_list), list_interp(S_1, F_v_headers, F_v_list)


def seismicity_level_get():
    """Not implemented"""
    pass


def code_period(h_n, s_type):
    """Returns period in seconds when given height (h_n) in metres and ASCE41-17 structural type (e.g. 'C1')"""
    return C_t_get(s_type) * (h_n / 0.3048) ** beta_get(s_type)


def S_a_func(T, S_s, S_1, site_class, T_L = 12.0, B_1 = 1.0):
    """Returns the spectral acceleration when provided with period (T), S_s, S_1 and site class"""
    F_a, F_v = site_coeffs(site_class, S_s, S_1)
    S_xs = F_a * S_s
    S_x1 = F_v * S_1
    T_s = S_x1 / S_xs
    T_0 = 0.2 * T_s
    return min(S_xs, S_xs*((5.0/B_1-2.0)*T/T_s + 0.4), S_x1 / B_1 / T) if T < T_L else T_L * S_x1 / (B_1*T**2.0)


# -----------------------------------------------------------------
# ----- Tier One Data and Functions -------------------------------
# -----------------------------------------------------------------

struct_types_dict = {
'W1': 'Light Wood Frame',
'W1A': 'Multistory Multi-Unit Residential Wood Frame',
'W2': 'Commercial and Industrial Wood Frame',
'S1': 'Steel MRF',
'S2': 'Steel Braced Frame',
'S2A': 'Steel Braced Frame Flexible Diaphragm',
'S3': 'Steel Light Frame',
'S4': 'Steel Frame RC SW',
'S5': 'Steel Frame URM INF',
'S5A': 'Steel Frame URM INF Flexible Diaphragm',
'C1': 'RC MRF',
'C2': 'RC Shear Wall',
'C2A': 'RC Shear Wall Flexible Diaphragm',
'C3': 'RC Frame URM Infill',
'C3A': 'RC Frame URM Infill Flexible Diaphragm',
'PC1': 'Precast/Tilt-Up RC SW',
'PC1A': 'Precast/Tilt-Up RC SW Flexible Diaphragm',
'PC2': 'Precast RC Frame with Shear Walls',
'PC2A': 'Precast RC Frame without Shear Walls',
'RM1': 'Reinforced Masonry Bearing Walls',
'RM2': 'Reinforced Masonry Bearing Walls Flexible Diaphragm',
'URM': 'Unreinforced Masonry Bearing Walls',
'URM': 'Unreinforced Masonry Bearing Walls Flexible Diaphragm'
}


# Table 10-1
struct_types_full_dict = {
'W1': {'Description':'Light Wood Frame', 'L':3, 'M':3, 'H':2},
'W1A': {'Description':'Multistory Multi-Unit Residential Wood Frame', 'L':3, 'M':3, 'H':2},
'W2': {'Description':'Commercial and Industrial Wood Frame', 'L':3, 'M':3, 'H':2},
'S1': {'Description':'Steel MRF', 'L':6, 'M':4, 'H':3},
'S1': {'Description':'Steel MRF', 'L':4, 'M':4, 'H':3},
'S2': {'Description':'Steel Braced Frame', 'L':6, 'M':4, 'H':3},
'S2A': {'Description':'Steel Braced Frame Flexible Diaphragm', 'L':3, 'M':3, 'H':3},
'S3': {'Description':'Steel Light Frame', 'L':2, 'M':2, 'H':2},
'S4': {'Description':'Steel Frame RC SW', 'L':6, 'M':4, 'H':3},
'S5': {'Description':'Steel Frame URM INF', 'L':3, 'M':3, 'H':0},
'S5A': {'Description':'Steel Frame URM INF Flexible Diaphragm', 'L':3, 'M':3, 'H':0},
'C1': {'Description':'RC MRF', 'L':3, 'M':0, 'H':0},
'C2': {'Description':'RC Shear Wall', 'L':6, 'M':4, 'H':3},
'C2A': {'Description':'RC Shear Wall Flexible Diaphragm', 'L':3, 'M':3, 'H':3},
'C3': {'Description':'RC Frame URM Infill', 'L':3, 'M':0, 'H':0},
'C3A': {'Description':'RC Frame URM Infill Flexible Diaphragm', 'L':3, 'M':0, 'H':0},
'PC1': {'Description':'Precast/Tilt-Up RC SW', 'L':3, 'M':2, 'H':2},
'PC1A': {'Description':'Precast/Tilt-Up RC SW Flexible Diaphragm', 'L':3, 'M':2, 'H':2},
'PC2': {'Description':'Precast RC Frame with Shear Walls', 'L':3, 'M':2, 'H':0},
'PC2A': {'Description':'Precast RC Frame without Shear Walls', 'L':0, 'M':0, 'H':0},
'RM1': {'Description':'Reinforced Masonry Bearing Walls', 'L':3, 'M':3, 'H':3},
'RM2': {'Description':'Reinforced Masonry Bearing Walls Flexible Diaphragm', 'L':6, 'M':4, 'H':3},
'URM': {'Description':'Unreinforced Masonry Bearing Walls', 'L':3, 'M':3, 'H':2},
'URMA': {'Description':'Unreinforced Masonry Bearing Walls Flexible Diaphragm', 'L':3, 'M':3, 'H':2}
}



M_s_dict = {'C1': {'CP':2.0, 'LS': 1.5, 'IO': 1.0},
            'C2': {'CP':4.5, 'LS': 3.0, 'IO': 1.5},
            'URM': {'CP':1.75, 'LS': 1.25, 'IO': 1.0}}


def M_s_func(name, perf):
    """Returns data from a dictionary of M_s
    inputs:
        name - the name of the system (currently a choice of 'C1', 'C2', 'URM')
        perf - performance level:
            'IO' - Immediate Occupancy
            'LS' - Life Safety
            'CP' - Collapse Prevention
    """
    return M_s_dict.get(name).get(perf)


table_4_2_headers = ('Start Year', 'End Year', 'Beams', 'Slabs and Columns', 'Walls')
table_4_2_ksi = ((1900, 1919, 2.0, 1.5, 1.0),
             (1920, 1949, 2.0, 2.0, 2.0),
             (1950, 1969, 3.0, 3.0, 2.5),
             (1970, 3000, 3.0, 3.0, 3.0))

table_4_3_headers = ('Start Year', 'End Year', 33, 40, 50, 60, 65, 70, 75)
table_4_2 = ((1911, 1959, 1, 1, 1, 0, 1, 0, 0),
             (1960, 1966, 1, 1, 1, 1, 1, 1, 1),
             (1967, 1987, 0, 1, 1, 1, 1, 1, 0),
             (1988, 3000, 0, 1, 1, 1, 1, 1, 1))


def v_j_avg_mf(n_c, n_f, V_j, A_c, M_s):
    """Returns 
    Inputs:
        n_c - number of columns
        n_f - number of frames in direction of loading
        V_j - storey shear (sect 4.4.2.2)
        A_c - sum of cross-sectional area of """
    return n_c/(n_c - n_f) * V_j / A_c / M_s


def D_r(k_b, k_c, h, E, V_c):
    """k_c = I_c / L_c
    NB This should be doubled if base of columns is pinned
    """
    return V_c * (k_b + k_c) / k_b / k_c * h / 12 / E


def v_j_avg_sw(V_j, A_w, M_s):
    return V_j / A_w / M_s


####### Plotting #########################################################
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def ASCE41_17_plots(S_s, S_1, T_code_s = 0.0, soil_classes = ['A', 'B', 'C', 'D', 'E'], S_a_func = S_a_func, site_class='B', **kwargs):
    """
    """
    T_max = max(3.0, 1.2 * T_code_s)
    lim = round(T_max * 100) + 1
    T_range = [0.01 * float(T) for T in range(1,lim)]

    #soil_classes = ['A', 'B', 'C', 'D', 'E']
    S_a_ranges = [[S_a_func(T, S_s, S_1, s_c) for T in T_range] for s_c in soil_classes]
    S_v_ranges = [[9.81 * 0.5 / pi * T * S_a for T, S_a in zip(T_range, S_a_range)] for S_a_range in S_a_ranges]
    S_d_ranges = [[0.5 / pi * T * S_v for T, S_v in zip(T_range, S_v_range)] for S_v_range in S_v_ranges]

    figs, (ax1,ax2,ax3, ax4) = plt.subplots(4,figsize = (10,15))

    [ax1.plot(T_range, S_a_range, label=s_c) for s_c, S_a_range in zip(soil_classes, S_a_ranges)]
    ax1.grid(True)
    ax1.legend(soil_classes)

    [ax2.plot(T_range, S_v_range, label=s_c) for s_c, S_v_range in zip(soil_classes, S_v_ranges)]
    ax2.grid(True)
    ax2.legend(soil_classes)

    [ax3.plot(T_range, S_d_range, label=s_c) for s_c, S_d_range in zip(soil_classes, S_d_ranges)]
    
    ax3.grid(True)
    ax3.legend(soil_classes)

    [ax4.plot(S_d_range, S_a_range, label=s_c) for s_c, S_d_range, S_a_range in zip(soil_classes, S_d_ranges, S_a_ranges)]
    
    ax4.grid(True)
    ax4.legend(soil_classes)
    
    if T_code_s > 0 and site_class in soil_classes:
        S_a_g = S_a_func(T_code_s, S_s, S_1, site_class)
        ax1.plot(T_code_s, S_a_g, marker = 'o')
        S_v_g = 0.5 / pi * T_code_s *  S_a_g * 9.81  # in m/s
        ax2.plot(T_code_s, S_v_g, marker = 'o')
        S_d_g = 0.5 / pi * T_code_s *  S_v_g  # in m
        ax3.plot(T_code_s, S_d_g, marker = 'o')
        ax4.plot(S_d_g, S_a_g, marker = 'o')
    
    return figs


def ASCE41_17_tripartite_plot(S_s, S_1, T_max = 3.0, soil_classes = ['A', 'B', 'C', 'D', 'E'], S_a_func = S_a_func, T_code_s = 0.0, site_class = 'B', **kwargs):
    """Returns a tripartite plot of ASCE41_17 spectra or some other specified function"""
    lim = round(T_max * 100) + 1
    T_range = [0.01 * float(T) for T in range(1,lim) if 0.01 * float(T) >= 0.1]

    S_a_ranges = [[S_a_func(T, S_s, S_1, s_c) for T in T_range] for s_c in soil_classes]
    
    plot_pts = []
    if T_code_s > 0 and site_class in soil_classes:
        S_a_g = S_a_func(T_code_s, S_s, S_1, site_class)
        S_v_g = 0.5 / pi * T_code_s *  S_a_g * 9.81  # in m/s
        plot_pts.append((T_code_s, S_v_g))
    
    return tripartite_plot(T_range, S_a_ranges, soil_classes, plot_pts=plot_pts)


def tripartite_plot(T_range, S_a_ranges, titles, plot_pts=[], **kwargs):
    """Plots tripartite data
        majorlines is a list of the minor log lines to plot (default [1,2,3,4,5,6,7,8,9])
    
    Args:
        T_range (list[float]: time values - it is assumed that the range is the same for all plots
        S_a_ranges: spectral acceleration (assumed to be in units of g)
        titles (list[str]): one title for each S_a_range
        plot_pts (list[tuple,list]): (T, S_v) points for plotting 
        kwargs: parameters for passing through to? Nowhere at present 
            (can specify figsize and majorlines)
    """
    # Check that data dimensions match
    err_msgs = []
    if not (len(S_a_ranges) == len(titles)):
        err_msgs.append(f'Number of curves does not match: len(S_a_ranges) = {len(S_a_ranges)}, len(titles) = {len(titles)}')
    S_a_ranges_len = [len(S_a_range) for S_a_range in S_a_ranges]
    if not (all(S_a_range_len == len(T_range) for S_a_range_len in S_a_ranges_len)):
        err_msgs.append(f'Number of curves does not match: len(T_range) = {len(T_range)}, len(S_a_ranges) = {S_a_ranges_len}')
    if len(err_msgs) > 0:
        err_msg = '\n'.join(err_msgs)
        raise ValueError(err_msg)
    
    # calculate the spectral pseudo velocities
    S_v_ranges = [[9.81 * 0.5 / pi * T * S_a for T, S_a in zip(T_range, S_a_range)] for S_a_range in S_a_ranges]
    
    figsize = kwargs.get('figsize', (10,8))
    fig0, ax0 = plt.subplots(figsize = figsize)
    [ax0.loglog(T_range, S_v_range, label=s_c) for s_c, S_v_range in zip(titles, S_v_ranges)]
    # ax0.loglog(T_code_s, S_v_g, marker = 'o') # marking the results for the building period

    ax0.grid(True)
    #fig = plt.gcf(); fig.set_size_inches(16,8)
    ax0.legend(titles)

    ##### Tripartite Lines ###########
    x0 = fig0.axes[0].get_xlim() #; print(x0)
    y0 = fig0.axes[0].get_ylim() #; print(y0)
    
    T_min, T_max = x0  
    log_T_min, log_T_max = log10(T_min), log10(T_max)
    #print('T_min, T_max', T_min, T_max)
    
    v_min, v_max = y0  
    log_v_min, log_v_max = log10(v_min), log10(v_max)
    #print('v_min, v_max', v_min, v_max)
    
    a_min, a_max = [2.0 * pi / T * v for v, T in ((v_min, T_max),(v_max, T_min))]
    log_a_min, log_a_max = log10(a_min), log10(a_max)
    #print('a_min, a_max', a_min, a_max)
    
    d_min, d_max = [0.5 / pi * T * v for v, T in ((v_min, T_min),(v_max, T_max))]
    log_d_min, log_d_max = log10(d_min), log10(d_max)
    #print('d_min, d_max', d_min, d_max)
    
    # Calculate a-values and d-values on the log-log line between a_min & a_max and d_min & d_max respectively
    major_lines = kwargs.get('major_lines', [1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    a_fac, a_unit = (1.0, 'm/s^2')
    #a_fac, a_unit = (9.81, 'g') # does not work!
    
    d_fac, d_unit = (1.0, 'm')
    
    #===========
    # generate the list of acceleration lines provided they are within the limits
    a_list = [(m * 10**float(n)) for n in range(int(log10(a_min))-2, int(log10(a_max))) for m in major_lines if (m * 10**float(n) * a_fac > a_min) and (m * 10**float(n) * a_fac < a_max)]
    a_prec = [max(0,int(1+log10(0.99/a))) for a in a_list]
    a_str = ['{:.{prec}f} {}'.format(a, a_unit, prec=p) for a, p in zip(a_list, a_prec)]
    a_values = [a * a_fac for a in a_list]
    #[print(a,b,s) for a,b,s in zip(a_list, a_values, a_str)]
    
    # generate the list of displacement lines provided they are within the limits
    d_list = [(m * 10**float(n)) for n in range(int(log10(d_min)), int(log10(d_max))+1) for m in major_lines if (m * 10**float(n) * d_fac > d_min) and (m * 10**float(n) / d_fac < d_max)]
    d_prec = [max(0,int(1+log10(0.99/d))) for d in d_list]
    d_str = ['{:.{prec}f} {}'.format(d, d_unit, prec=p) for d, p in zip(d_list, d_prec)]
    d_values = [d * d_fac for d in d_list]
    #print('d_values (min & max)', d_values[0], d_values[-1])
    # 
    log_a_values = [log10(a) for a in a_values]
    log_d_values = [log10(d) for d in d_values]

    #change the following lines
    #a_ls = ['-' if abs(a + log10(a_fac) - int(a + log10(a_fac))) < 0.001 else ':' for a in log_a_values]
    a_ls = ['-' if abs(log10(a) - int(log10(a))) < 0.001 else ':' for a in a_list]
    #d_ls = ['-' if abs(d - int(d)) < 0.001 else ':' for d in log_d_values]
    d_ls = ['-' if abs(log10(d) - int(log10(d))) < 0.001 else ':' for d in d_list]

    ta_values = [(a - log_a_min)/(log_a_max - log_a_min) for a in log_a_values]
    #print('ta_min, ta_max', ta_values[0], ta_values[-1])
    td_values = [(d - log_d_min)/(log_d_max - log_d_min) for d in log_d_values]
    #print('td_min, td_max', td_values[0], td_values[-1])

    va_values = [log_v_min + t * (log_v_max - log_v_min) for t in ta_values]
    #print('va_min, va_max', 10**va_values[0], 10**va_values[-1])
    vd_values = [log_v_min + t * (log_v_max - log_v_min) for t in td_values]
    #print('vd_min, vd_max', 10**vd_values[0], 10**vd_values[-1])

    Ta_values = [log_T_max + t * (log_T_min - log_T_max) for t in ta_values]
    #print('Ta_min, Ta_max', 10**Ta_values[0], 10**Ta_values[-1])
    Td_values = [log_T_min + t * (log_T_max - log_T_min) for t in td_values]
    #print('Td_min, Td_max', 10**Td_values[0], 10**Td_values[-1])

    #s1
    s_a2 = [min(log_v_max - v, log_T_max - T) for v, T in zip(va_values, Ta_values)]
    s_a1 = [max(log_v_min - v, log_T_min - T) for v, T in zip(va_values, Ta_values)]
    #print('s_a1[0], s_a2[0]', s_a1[0], s_a2[0])

    # identifying the index of the longest acceleration line
    (m_a, i_a) = max((v2-v1,i) for i, (v1, v2, als) in enumerate(zip(s_a1, s_a2, a_ls)) if als == '-')
    #print('m_a, i_a', m_a, i_a)

    #s1
    s_d2 = [max(v - log_v_max, log_T_min - T) for v, T in zip(vd_values, Td_values)]
    s_d1 = [min(v - log_v_min, log_T_max - T) for v, T in zip(vd_values, Td_values)]
    #print('s_d1[0], s_d2[0]', s_d1[0], s_d2[0])

    # identifying the index of the longest displacement line
    (m_d, i_d) = max((v1-v2,i) for i, (v1, v2, dls) in enumerate(zip(s_d1, s_d2, d_ls)) if dls == '-')
    #print('m_d, i_d', m_d, i_d)

    # calculate the coordinates of the ends of the acceleration lines
    a_coords = [((10**(T+s1), 10**(T+s2)), (10**(v+s1), 10**(v+s2))) for v, T, s1, s2 in zip(va_values, Ta_values, s_a1, s_a2)]
    a_lines = [lines.Line2D(P1, P2, transform=ax0.transData, axes=ax0, color='grey', ls = ls, lw = 0.5, zorder=0) for (P1, P2), ls in zip(a_coords, a_ls)]

    # calculate the coordinates of the ends of the displacement lines
    d_coords = [((10**(T+s1), 10**(T+s2)), (10**(v-s1), 10**(v-s2))) for v, T, s1, s2 in zip(vd_values, Td_values, s_d1, s_d2)]
    d_lines = [lines.Line2D(P1, P2, transform=ax0.transData, axes=ax0, color='grey', ls = ls, lw = 0.5, zorder=0) for (P1, P2), ls in zip(d_coords, d_ls)]

    #print('a_coords', a_coords[i_a])
    #print('d_coords', d_coords[i_d])

    ax_lines = [lines.Line2D(P1, P2, transform=ax0.transData, axes=ax0, color='magenta', ls = '-', lw = 1.0, zorder=0) for (P1, P2) in [a_coords[i_a], d_coords[i_d]]]

    ax0.lines.extend(a_lines + d_lines + ax_lines) #, l3, l4])

    #ax0.text(0.5*(T_min + T_max), 0.5*(v_min + v_max), "text on plot $m/s^2$")
    [ax0.text(10**T, 10**v, '$'+txt+'$', rotation=45, zorder=0.1) for T, v, txt in zip(Ta_values, va_values, a_str)]
    [ax0.text(10**T, 10**v, '$'+txt+'$', rotation=-45, zorder=0.1) for T, v, txt in zip(Td_values, vd_values, d_str)]
    
    # plot any additionally provided points
    if len(plot_pts) > 0:
        [ax0.plot(T, v, marker = 'o') for T,v in plot_pts]
    
    return fig0


