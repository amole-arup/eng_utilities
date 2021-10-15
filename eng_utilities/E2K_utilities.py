"""A set of utilities that act on ETABS-specific data structures
"""

from geometry_utilities import *


def get_ETABS_local_axes(line, angle=0):
    """Returns the local axes based on ETABS default definitions
    
    Note that ETABS defaults are different from GSA.
    Default tolerance for verticality in ETABS is the sine of the angle
    to be one thousandth (10^-3). GSA has a default verticality of one degree. 
    To set the GSA tolerance to be the same as ETABS, set 
    vertical_angle_tolerance = `asin(0.001)*180/pi` = 0.057296 deg
    
    Args:
        line: a 2-tuple of 3-tuples representing the start and end nodes in 
            global cartesian coordinates
        angle (float): rotation in degrees as positive rotation about axis 1.
        vertical_angle_tolerance (float): angle tolerance in degrees
    
    Returns:
        a 3-tuple of 3-tuples representing the local unit coordinates (x,y,z) in 
            global cartesian coordinates
    
    >>> fmt_3x3(get_ETABS_local_axes(((0,0,0),(0,0,3.5)),0),'7.4f')
    '( 0.0000,  0.0000,  1.0000), ( 1.0000,  0.0000,  0.0000), ( 0.0000,  1.0000,  0.0000)'
    >>> fmt_3x3(get_ETABS_local_axes(((0,0,0),(0,0,3.5)),30),'7.4f')
    '( 0.0000,  0.0000,  1.0000), ( 0.8660,  0.5000,  0.0000), (-0.5000,  0.8660,  0.0000)'
    >>> fmt_3x3(get_ETABS_local_axes(((8.5,0,3.5),(8.5,1,3.5)), 30),'7.4f')
    '( 0.0000,  1.0000,  0.0000), ( 0.5000,  0.0000,  0.8660), ( 0.8660,  0.0000, -0.5000)'
    >>> fmt_3x3(get_ETABS_local_axes(((0,0,0),(7.2,9.0,3.5)),30),'7.4f')
    '( 0.5977,  0.7472,  0.2906), ( 0.2332, -0.5088,  0.8287), ( 0.7670, -0.4276, -0.4784)'
    """
    #vert = True
    vector = sub3D(line[1], line[0])
    dir1 = unit3D(vector)
    dir2 = (0,0,0)
    dir3 = (0,0,0)
    #length = mag3D(vector)
    ang_rad = angle * pi / 180
    
    if sin3D(vector, (0, 0, 1)) < 0.001: # Column
        #vert = True
        dir1 = (0, 0, 1)
        dir2 = (1, 0, 0) if angle == 0 else (cos(ang_rad), sin(ang_rad), 0)
        dir3 = (0, 1, 0) if angle == 0 else (-sin(ang_rad), cos(ang_rad), 0)
    else:  # Not a column
        #vert = False
        dir3_ = unit3D(cross3D(dir1, (0,0,1)))
        dir2_ = unit3D(cross3D(dir3_, dir1))
        dir2 = dir2_ if angle == 0 else rotQ(dir2_, ang_rad, dir1)
        dir3 = dir3_ if angle == 0 else rotQ(dir3_, ang_rad, dir1)
    return dir1, dir2, dir3

