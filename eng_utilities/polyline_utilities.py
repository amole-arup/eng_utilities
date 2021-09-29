""""""

from eng_utilities.geometry_utilities import *

def bounding_box(coords):
    """Runs through a collection of x,y tuple pairs and 
    extracts the values (xmin,ymin),(xmax,ymax)."""
    xmin = coords[0][0]  ;   xmax = coords[0][0]
    ymin = coords[0][1]  ;   ymax = coords[0][1]
    for xy in coords[1:]:
        x, y = xy
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    return [(xmin, ymin),(xmax, ymax)]


def perim_area_centroid(perim):
    """Calculates the area and centroid of sections defined by 
    closed 2D polylines (a list of 2D tuples - 
    e.g. [(x1, y1), (x2, y2), ...])"""
    res = [0.0, 0.0, 0.0]
    x1, y1 = perim[0]
    # close the polyline if necessary
    if perim[0] != perim[-1]: perim.append(perim[0])
    for p2 in perim[1:]:
        x2, y2 = p2
        area = (x1 * y2 - x2 * y1) / 2.0
        res[0] += area
        res[1] += (x1 + x2) * area / 3.0
        res[2] += (y1 + y2) * area / 3.0
        x1, y1 = x2, y2
    return (0.0,(0.0,0.0)) if res[0] == 0.0 else (res[0], (res[1]/res[0],res[2]/res[0]))


def perim_props(pt_list):
    """Calculates the geometric properties of sections defined by closed 2D polylines (FullSecProp2D).
    Inputs is: 
        pt_list - a polyline defined by a list of points as 2D tuples
    """
    if pt_list[0] != pt_list[-1]:
        pt_list = list(pt_list)
        pt_list.append(pt_list[0])
    
    p_length = sum(dist2D(pt1, pt2) for pt1, pt2 in zip(pt_list[:-1], pt_list[1:]))
        
    res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x1, y1 = pt_list[0]
    for p2 in pt_list[1:]:
        x2, y2 = p2
        area = (x1 * y2 - x2 * y1) / 2.0
        res[0] += area
        res[1] += (x1 + x2) * area / 3.0
        res[2] += (y1 + y2) * area / 3.0
        res[3] += (y1**2.0 + y1*y2 + y2**2.0) * area / 6.0
        res[4] += (x1**2.0 + x1*x2 + x2**2.0) * area / 6.0
        res[5] += (x1*y2 + 2.0*x1*y1 + 2.0*x2*y2 + x2*y1) * area / 12.0
        x1, y1 = x2, y2
    if res[0] == 0:
        err_msg = 'Polyline has zero area: \n' + str(pt_list)
        return {'A': 0, 'Error Message' : err_msg}
    else:
        return {'P': p_length, 'A': res[0], 'Cy': res[1]/res[0], 'Cz': res[2]/res[0], 
            'Iyy': res[3], 'Izz': res[4], 'Iyz': res[5]}            


def perim_full_props(pt_list):
    """Calculates the geometric properties of sections defined by closed 2D polylines (FullSecProp2D).
    Inputs are:
        pt_list - a polyline defined by a list of points as 2D tuples
        centered - if True returns results for section centered at centroid (default is False)
        as_dict - if True returns results in the form of a dictionary (default is False)
    Results are:
        About origin   - A, Cy, Cz, Iyy0, Izz0, Iyz0
        About centroid  - Iyy, Izz, Iyz, 
        Principal Axes - Iuu, Ivv, theta_rad
        Elastic Moduli - Zzmax, Zzmin, Zymax,  Zymin
    Note:
        To rotate to principal coordinates rotate by negative theta_rad.
    """
    if pt_list[0] != pt_list[-1]:
        pt_list = list(pt_list)
        pt_list.append(pt_list[0])
    
    props = perim_props(pt_list)
    if props['A'] == 0:
        err_msg = props['Error Message'] + '\n'
        return {'A': 0, 'Error Message' : err_msg}
    
    p_length, area, c_y, c_z, I_yy, I_zz, I_yz = [props.get(p,0) for p in 
                                        ('P', 'A', 'Cy', 'Cz', 'Iyy', 'Izz', 'Iyz')]
    pt_list_c = [[y - c_y, z - c_z] for y, z in pt_list]
    (ymin, zmin),(ymax, zmax) = bounding_box(pt_list_c[:-1])
    props2 = perim_props(pt_list_c)
    if props2['A'] == 0:
        err_msg = props2['Error Message'] + '\n' + str(pt_list)
        return {'A': 0, 'Error Message' : err_msg}
    
    area2, c_y2, c_z2, I_yy2, I_zz2, I_yz2 = [props2.get(p,0) for p in 
                                              ('A', 'Cy', 'Cz', 'Iyy', 'Izz', 'Iyz')]
    # c_y2 and c_z2 should be zero
    theta = 0.5 * atan2(-2.0 * I_yz2, I_yy2 - I_zz2)
    I_ave = 0.5 * (I_yy2 + I_zz2)
    # `radius` refers to the Mohr's circle in the I, Ixy space.
    radius = ((0.5 * (I_yy2 - I_zz2)) ** 2.0 + I_yz2 ** 2.0) ** 0.5
    return {'P': p_length, 'A': area, 'Cy': c_y, 'Cz': c_z, \
            'Iyy0': I_yy, 'Izz0': I_zz, 'Iyz0': I_yz, \
            'Iyy': I_yy2, 'Izz': I_zz2, 'Iyz': I_yz2, \
            'Iuu': I_ave - radius, 'Ivv': I_ave + radius, 'theta_rad': theta, \
            'zmax': zmax, 'zmin': zmin, 'ymax': ymax, 'ymin': ymin, \
            'Zzmax': I_yy2 / zmax, 'Zzmin': I_yy2 / -zmin, \
            'Zymax': I_zz2 / ymax, 'Zymin': I_zz2 / -ymin}


def perim_rotate2D(perim, ang):
    return [rotate2D(pt, ang) for pt in perim]


def sec_area_3D(pt3dlist):
    """Calculates the area of sections defined by
    closed coplanar polylines on any plane (does not need 
    to match orthogonal planes).
    Polylines should be defined as a list of triplets.
    Note that there is no check if points are not coplanar 
    or if the lines cross."""
    # close the polyline if necessary
    if pt3dlist[0] != pt3dlist[-1]: pt3dlist.append(pt3dlist[0])
    v_sum = [0.0, 0.0, 0.0]
    pt1 = pt3dlist[0]
    # Move the coordinates so that the first is at 0,0,0
    vlist = [sub3D(pt,pt1) for pt in pt3dlist[1:-1]]
    v1 = vlist[0]
    # Iterate over the vectors
    for v2 in vlist[1:]:
        v_cross = cross3D(v1,v2)
        v_sum = add3D(v_sum, v_cross)
        v1 = v2    
    # v_sum is a normal vector. The area is half the magnitude.
    area  = 0.5 * mag3D(v_sum)
    #print("So the Area of Slab is ..."+ str(area) )
    return area # This will always be positive
