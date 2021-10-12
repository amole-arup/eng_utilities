"""A set of functions for lines and polylines that are defined 
as tuples of numerical tuples. Lines are assumed to be 2-ples of n-ples
There are also some functions that can include alphabetic keys that 
may be present as keys in dictionaries or as elements at the end of the tuple. 

These include the following:
* Properties for closed polylines
* Identifying loops of lines (2-ples of n-ples)
* Finding crossing lines

TODO
- intersection2D - add checks for parallel elements - identify
    whether they are collinear and if so, do they overlap?
- Sort out routine for identifying intersections in the XY plane
    which is for finding secondary beams intersecting with
    primary beams and must therefore handle IDs
- Make sure that docstrings are consistent with Google
    standards for automatic documentation
- add tests
"""

from eng_utilities.geometry_utilities import *
from collections import namedtuple, OrderedDict
from operator import le, lt

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
    x1, y1, *_ = perim[0]
    # close the polyline if necessary
    if perim[0] != perim[-1]: perim.append(perim[0])
    for p2 in perim[1:]:
        x2, y2, *_ = p2
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
    """Calculates the geometric properties of sections defined by closed 2D polylines 
    (FullSecProp2D).

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


def min_x_pt(pt_dict):
    """Returns the key of the point at minimum x 
    (with minimum y) in the XY plane (Z is ignored)
    pt_dict is a collection of 'pt_ID: (x, y, z)'
    key-value pairs"""
    nmin = (9.9E12, 9.9E12)
    the_node = None
    for k,v in pt_dict.items():
        if v[0] == nmin[0]:
            if v[1] < nmin[1]:
                the_node = k
                nmin = (v[0], v[1])
        elif v[0] < nmin[0]:
            the_node = k
            nmin = (v[0], v[1])
    return the_node


def loop_finder(pt_dict, connections_dict, start_pt_ID=None, print_points=False):
    """Returns a list containing 
    a closed loop of connected points
    
    The strategy is to follow connected nodes, always 
    choosing the one that is on the right hand side
    so that the final loop is anti-clockwise. Note
    that the returned list may not be the only loop 
    present. Note also that this algorithm does not
    check for crossing lines.

    Args:
        pt_dict:  is a collection of 'pt_ID: (x, y, z)'
            key-value pairs
        connections_dict: is a collection of 
            'pt0_ID: [pt1_ID, pt2_ID, pt3_ID, ...]'
            where pt0 is connected to each of the 
            referenced points pt1_ID, pt2_ID etc.
        start_pt_ID: """
    
    if start_pt_ID is not None:
        pt0_ID = start_pt_ID
    else:
        pt0_ID = min_x_pt(pt_dict)
    
    # do not shrink connections_dict to limit it to nodes in pt_dict 

    if print_points:
        #print('input types:', type(pt_dict), type(connections_dict))
        #print('connections_dict:\n', connections_dict)
        pt_set = set(pt_dict.keys())
        conn_set = set(sum([[k] + list(v) for k, v in connections_dict.items()],[]))
        #print('Points in connections_dict', [[k] + list(v) for k, v in connections_dict.items()])
        print('# Number of points missing from pt_dict:', len(conn_set - pt_set))
        
    theta = 0
    
    loop_ID_list = [pt0_ID]
    old_ID = None
    pt_ID = pt0_ID
    while True:
        # This is a process that walks through the node network until it gets
        # back to the start (which is why it is a 'while', not a 'for').
        connected_node_IDs = connections_dict.get(pt_ID, [])
        node_IDs = [node_ID for node_ID in connected_node_IDs if node_ID != old_ID]
        node_IDs = [node_ID for node_ID in node_IDs if node_ID in pt_dict]
        pt_coords = pt_dict.get(pt_ID)
        if len(node_IDs) == 0 and pt_ID == pt0_ID:
            loop_ID_list = [pt0_ID]
            break
        
        if len(node_IDs) == 0 and old_ID in connected_node_IDs:
            new_pt_ID = old_ID # to allow it to return from a branch
            theta = angfix(theta + pi) # 180deg reversed
        else:
            # note that this does not currently handle situations where beams have zero length
            try:
                node_coords = [subNDx(pt_dict.get(node_ID), pt_coords) for node_ID in node_IDs] # relative vectors
            except:
                print(node_IDs, '\n', [pt_dict.get(node_ID) for node_ID in node_IDs], '\n', pt_coords)
            node_coords = [subNDx(pt_dict.get(node_ID), pt_coords) for node_ID in node_IDs] # relative vectors
            # convert to polar coordinates (assigning high angle if magnitude is zero)
            polar_coords = [cart2cyl(coords, 9 * pi) for coords in node_coords] # polar coordinates
            angles = [ang for _, ang, _ in polar_coords] # extract angles
            # convert to relative angles (assigning high angle to case already assigned high angle)
            rel_angles = [(angfix(ang - theta) if ang <= pi else (9 * pi)) for ang in angles] # angles relative to the incoming beam
            # choose pt with minimum relative angle (i.e. on the right side for anti-clockwise loop)
            min_ang_pt = min((rel_ang, ID, ang) for rel_ang, ID, ang in zip(rel_angles, node_IDs, angles))
            new_pt_ID = min_ang_pt[1]
            theta = min_ang_pt[2]
            if len(loop_ID_list) > 1:
                if new_pt_ID == loop_ID_list[1]:
                    break
        loop_ID_list.append(new_pt_ID)
        old_ID = pt_ID
        pt_ID = new_pt_ID
    return loop_ID_list


def line_intersection2D(line1, line2, is_inclusive=True, tol=0.01):
    """Returns a dictionary of any intersection when 
    provided with two lines (each defined as a pair of tuples).

    The intersection type is reported as:
        'parallel':  lines are parallel (no intersection)
        'neither' :  intersection is outside both lines
        'line1'   : intersection is only inside the extent of line1
        'line2'   :  intersection is only inside the extent of line2
        'both'    : intersection occurs within both line extents

    Intersections are defined in the XY plane. Other coordinates
    will be carried over, but ignored. If the is_inclusive option
    is maintained, crossing definition will include direct 
    intersections at line ends.

    Lines are defined as a tuple of tuples:
        ((x1, y1), (x2, y2))
    These are then converted into parametric format:
        point + t * vector
    pt(t) = p + t * v
    p1(t1) = p1 + t1 * v1
    p2(t2) = p2 + t2 * v2
    
    Result: 

    """
    op = le if is_inclusive else lt

    p1, v1 = line1[0], subNDx(line1[1], line1[0])
    p2, v2 = line2[0], subNDx(line2[1], line2[0])
     
    denom = v1[0] * v2[1] - v1[1] * v2[0]
    if denom == 0:
        return {'type': 'parallel'}
    else:
        t1 = ((p1[1] - p2[1]) * v2[0] - (p1[0] - p2[0]) * v2[1]) / denom        
        t2 = ((p1[1] - p2[1]) * v1[0] - (p1[0] - p2[0]) * v1[1]) / denom
        
        crossing_type = None
        if op(0, t1) and op(t1, 1):
            crossing_type = 'line1'
        if op(0, t2) and op(t2, 1):
            crossing_type = 'both' if crossing_type else 'line2'
        if crossing_type is None:
            crossing_type = 'neither'
        
        
        touching = []
        if (-tol <= t2 <= tol):
            touching.append('line2_start')
        if ((1 - tol) <= t2 <= (1 + tol)):
            touching.append('line2_end')
        if (-tol <= t1 <= tol):
            touching.append('line1_start')
        if ((1 - tol) <= t1 <= (1 + tol)):
            touching.append('line1_end')
        return {'intersection': addND(p1, scaleND(v1, t1)), 
        't1': t1, 't2': t2, 'type': crossing_type, 'touching': touching,
        }
    

def self_intersections(line_list, is_sorted=False, is_inclusive=False, tol = 0.01):
    """Calculates self intersections using a sweep algorithm
    along the x-axis.
    
    line_list is a list of lines defined as pairs of tuples"""
    if is_sorted == False:
        sorted_list = sorted((pt1, pt2, *tail) if pt1[0] < pt2[0] else (pt2, pt1, *tail) 
            for pt1, pt2, *tail in line_list)
    else:
        sorted_list = list(line_list)
    
    line_stack = []
    cross_list = []

    for line in sorted_list:
        x = line[0][0]
        next_stack = []
        for line2 in line_stack:
            if line2[1][0] >= x:
                next_stack.append(line2)
                crossing = line_intersection2D(line, line2, is_inclusive, tol=tol)
                if crossing['type'] == 'both':
                    cross_list.append(crossing['intersection'])
        line_stack = next_stack.copy()
        line_stack.append(line)
    return cross_list


def line_interpolate_y(line, x):
    """Interpolates y values on a line when given x
    To avoid duplicate hits for nodes level with the 
    joints between lines, the end of a line is not 
    considered an intersection."""
    if line[0][0] == x:
        return line[0][1]
    #if line[1][0] == x:
    #    return line[1][1]
    elif (line[0][0] < x < line[1][0]) and ((line[1][0] - line[0][0]) != 0):
        return line[0][1] + (x - line[0][0]) * (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
    else:
        return None


def nodes_outside_loop(node_coord_dict, pt_loop):
    """Returns a dictionary containing the list of nodes 
    outside and inside a loop
    node_coord_dict = {'N1': (5,8,1), 'N2': (9,-1,4), ...}
    pt_loop = ['N1', 'N4', ..., 'N1']
    Note:
        1. that node_coord_dict is the list of nodes considered
        2. that the loop must be closed"""
    
    coord_loop = [node_coord_dict.get(pt) for pt in pt_loop]
    loop_xmin = min(x for x, *_ in coord_loop)
    loop_xmax = max(x for x, *_ in coord_loop)
    #print('x_minmax', loop_xmin, loop_xmax)

    line_loop = [(pt1, pt2) for pt1, pt2 
            in zip(coord_loop[:-1], coord_loop[1:])]
    
    # Sort the lines and points based on the x-axis
    sorted_line_list = sorted((pt1, pt2) if pt1[0] < pt2[0] else (pt2, pt1) 
            for pt1, pt2 in line_loop)
    
    sorted_point_list = sorted((coords[0], coords[1], ID) 
            for ID, coords in node_coord_dict.items())
    
    #print('sorted_point_list', sorted_point_list)
    #print()
    #print('sorted_line_list', sorted_line_list)

    line_gen = (line for line in sorted_line_list)
    #point_gen = (point for point in sorted_point_list)

    line = next(line_gen)
    line_stack = []
    inside_nodes = []
    outside_nodes = []
    # iterate over the nodes in the list
    for node in sorted_point_list:
        #print('____\nnode', node)
        node_x, node_y, nodeID = node
        
        # No need to check nodes that are below 
        # or above the loop
        if node_x < loop_xmin:
            outside_nodes.append(nodeID)
            continue
        elif node_x > loop_xmax:
            outside_nodes.append(nodeID)
            continue
        
        if node[2] in pt_loop:  # ignore nodes in loop 
            continue
        
        # tidy stack - only keep lines with end_x > node_x
        if len(line_stack) > 0:
            line_stack = [line for line in line_stack if line[1][0] > node_x]
        
        while line is not None:
            # try adding lines to line_stack
            # check whether line is ahead of node 
            # and whether line is vertical
            start_x = line[0][0]
            end_x = line[1][0]
            if (start_x <= node_x <= end_x) and (start_x != end_x):
                line_stack.append(line)
            elif start_x > node_x: # 
                break # do not replace, wait
            
            try:     # get new line
                line = next(line_gen)
            except StopIteration:
                line = None
                break
        
        """print(node)
        print('\nline_stack')
        [print(line) for line in line_stack]
        print()"""

        # check node against limits - how many crossings
        #print(f'Check {nodeID}')
        y_values = [line_interpolate_y(line, node_x) for line in line_stack]
        count_above_node_y = sum(1 for y in y_values  if ((y is not None) and (y >= node_y)))
        """if nodeID in [(86, 'ROOF'), (89, 'ROOF'), (98, 'ROOF'), (101, 'ROOF')]:
            print(f'ID: {nodeID} ({node_x}, {node_y}), count: {count_above_node_y} | ')
            print('y-values = ', (node_y, y_values))
            print('line_stack = ', line_stack)"""
        if count_above_node_y % 2 == 0:
            outside_nodes.append(nodeID)
        else:
            inside_nodes.append(nodeID)
    
    #print('pt_loop')
    #[print(pt) for pt in zip(coord_loop, pt_loop)]
    return {'outside': outside_nodes, 'inside': inside_nodes}
        
    
def all_loops_finder(pt_dict, connections_dict, sort_points=True, print_points=False):
    """Returns a list containing 
    a closed loop of connected points"""

    first_point_ID = None

    # Sort the pt_dict according to x
    if sort_points:
        pt_dict = {k:v for v, k in sorted([(v, k) for k, v in pt_dict.items()])}
    
    loops_list = []

    i = 0
    while True:
        i += 1
        # 
        if len(pt_dict) == 0:
            break
        if sort_points:
            first_point_ID = list(pt_dict)[0]
        
        loop = loop_finder(pt_dict, connections_dict, first_point_ID, print_points=print_points)
        if print_points:
            print(f'# loop {i}: ', loop)
            print(f'loop_{i} = ', [pt_dict.get(pt) for pt in loop])

        if len(loop) == 1:
            pt_dict.pop(loop[0])
            if len(pt_dict) == 0:
                print(f'{i}: only one item in loop')
                break
        elif len(loop) > 1:
            inside_outside_nodes = nodes_outside_loop(pt_dict, loop)
            inside_nodes = inside_outside_nodes['inside']
            outside_nodes = inside_outside_nodes['outside']
            if print_points:
                #print(f'{i}: outside ', outside_nodes)
                print(f'outside_{i} = ', [pt_dict.get(pt) for pt in outside_nodes])
                #print(f'{i}: inside ', inside_nodes)
                print(f'inside_{i} = ', [pt_dict.get(pt) for pt in inside_nodes])
            loops_list.append(loop)
            if len(outside_nodes) > 0:                
                pt_dict = {k:v for k, v in pt_dict.items() if k in outside_nodes}
            else:
                break
        else:  # something has gone wrong
            if len(pt_dict) > 0:
                print(f'Something has gone wrong in loop finding...')
                print(f'{len(pt_dict)} nodes are still in pt_dict')
            break
    
    return loops_list


def main():
    lines2D = (
        ((6,3), (1,4), 'A'), ((2,2), (3,4), 'B'), 
        ((4,1), (0,3), 'C'), ((0,-1), (1,1), 'D'), 
        ((5,1), (7,2), 'E'), ((3,4), (5,5), 'F'), 
        ((1,0), (6,5), 'G')
        )
    lines2Da = (
        ((6,3,'a1'), (1,4,'a2'), 'A'), ((2,2,'b1'), (3,4,'b2'), 'B'), 
        ((4,1,'c1'), (0,3,'c2'), 'C'), ((0,-1,'d1'), (1,1,'d2'), 'D'), 
        ((5,1,'e1'), (7,2,'e2'), 'E'), ((3,4,'f1'), (5,5,'f2'), 'F'), 
        ((1,0,'g1'), (6,5,'g2'), 'G')
        )
    z = 2
    lines3D = [((line[0][0], line[0][1], z), 
                (line[1][0], line[1][1], z), 
                line[2]) 
                for line in lines2D]
    lines3Da = [((line[0][0], line[0][1], z, line[0][2]), 
                (line[1][0], line[1][1], z, line[1][2]), 
                line[2]) 
                for line in lines2Da]
    lines3Db = [(((line[0][0], line[0][1], z), line[0][2]), 
                ((line[1][0], line[1][1], z), line[1][2]), 
                line[2]) 
                for line in lines2Da]
    print('Lines3D:', lines3D)
    print(self_intersections(lines3D, is_inclusive=True))
    print('Lines3Da:', lines3Da)
    #print(self_intersections(lines3Da, is_inclusive=True))
    print('Lines3Db:', lines3Db)
    #print(self_intersections(lines3Da, is_inclusive=True))



if __name__ == "__main__":
    main()
