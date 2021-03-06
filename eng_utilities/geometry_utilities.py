"""Utilities for geometry defined by tuples or lists of numbers"""

from math import pi, sin, cos, tan, atan, atan2, asin, acos, exp, log, log10
from functools import reduce
from operator import mul

def rad_to_deg(ang):
    """Converts radians to degrees"""
    return ang * 180 / pi


def deg_to_rad(deg):
    """Converts degrees to radians"""
    return deg / 180 * pi


def dist2D(pt1, pt2):
    """Returns distance between two 2D points (as two 2-tuples)"""
    return ((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)**0.5


def mag2D(v2D):
    """Returns the magnitude of the vector"""
    return (v2D[0]**2 + v2D[1]**2)**0.5


def neg2D(v2D):
    """Returns the negative of the vector"""
    return -v2D[0], -v2D[1]


def unit2D(v, default=None):
    """Returns the unit 2D vector in the same direction 
    as the input 2D vector (as one 2-tuple)"""    
    if mag2D(v) == 0.0 :
        if default is None:
            raise ValueError('Vector should have non-zero magnitude')
        else:
            return default
    else:
        return scale2D(v, 1.0 / mag2D(v))


def ang2D(pt):
    """Returns the anti-clockwise angle of the 
    vector from the x-axis (in radians)"""
    return atan2(pt[1],pt[0])


def scale2D(v,scale):
    """Returns a scaled 2D vector"""
    return (v[0] * scale, v[1] * scale)


def add2D(v1,v2):
    """Vector addition of v1 to v2"""
    return (v1[0] + v2[0], v1[1] + v2[1])


def sub2D(v1,v2):
    """Vector subtraction of v2 from v1"""
    return (v1[0] - v2[0], v1[1] - v2[1])


def mul2D(v1,v2):
    """Elementwise multiplication of vector v1 with v2"""
    return (v1[0] * v2[0], v1[1] * v2[1])


def div2D(v1,v2):
    """Elementwise division of vector v1 by v2"""
    return (v1[0] / v2[0], v1[1] / v2[1])


def cross2D(v1,v2):
    """calculates the scalar cross product magnitude of two 2D vectors, v1 and v2"""
    return v1[0]*v2[1] - v1[1]*v2[0]


def outer2D(v1, v2):
    """Calculates the magnitude of the outer product of two 2D vectors, v1 and v2"""
    return v1[0]*v2[1] - v1[1]*v2[0]


def dot2D(v1,v2):
    """calculates the scalar dot product of two 2D vectors, v1 and v2"""
    return v1[0]*v2[0] + v1[1]*v2[1]


def inner2D(v1, v2):
    """Calculates the inner product of two 2D vectors, v1 and v2"""
    return v1[0]*v2[0] + v1[1]*v2[1]


def sin2D(v1, v2):
    """calculates the sine of the angle between two 2D vectors, v1 and v2
    using the magnitude of the cross product of the normalised vectors."""    
    if mag2D(v1) * mag2D(v2) == 0.0 :
        raise ValueError('Both vectors should have non-zero magnitude')
    else:
        scale = 1 / mag2D(v1) / mag2D(v2)
        return cross2D(v1, v2) * scale


def cos_sim2D(v1, v2, default=None):
    """returns the cosine of the angle between two vectors (v1, v2) based on the dot product
    v1 . v2 = |v1|  |v2| cos (<v1,v2>)
    A result of 1 means they are parallel, -1 that they are anti-parallel,
    and 0 that they are perpendicular."""
    if mag2D(v1) * mag2D(v2) == 0.0 :
        #raise ValueError('Both vectors should have non-zero magnitude')
        if default is None:
            raise ValueError('Both vectors should have non-zero magnitude')
        else:
            return default
    else:
        return inner2D(v1, v2) / mag2D(v1) / mag2D(v2)


#def unit(self, magn = 1.0):
#    """Returns the unit vector (optionally scaled by a factor 'magn')"""
#    fac = magn / mag(self)
#    return (self[0] * fac, self[1] * fac)


def mid2D(p1,p2,h=0.5):
    """returns a point between the two points (h is position parameter defaulting to mid-point)"""
    return h * p1[0] + (1 - h) * p2[0]


def dev2D(v1,v2):
    """Provides the perpendicular distance of the end of line v1 from line v2"""
    v2mag = (v2[0]**2.0 + v2[1]**2.0)**0.5
    return 0 if v2mag == 0 else cross2D(v1,v2) / v2mag


def ave2D(v1, v2):
    """Returns the midpoint between to points (average)"""
    return 0.5 * (v1[0] + v2[0]), 0.5 * (v1[1] + v2[1])


def rotate2D(pt, ang):
    """Rotates a point (x, y) about the origin by an angle in radians"""
    return mag2D(pt) * cos(ang + ang2D(pt)), mag2D(pt) * sin(ang + ang2D(pt))


def planar_angle2D(v1, v2):
    """returns the angle of one vector relative to the other in the 
    plane defined by the normal (default is in the XY plane)
    NB This algorithm avoids carrying out a coordinate transformation 
    of both vectors. However, it only works if both vectors are in that 
    plane to start with. """
    return atan2(sin2D(v1, v2), cos_sim2D(v1, v2))


def bbox2D(coords):
    """Runs through x,y tuple triplets and extracts the values corresponding to
        'pt1': (xmin,ymin,zmin) - minimum corner
        'pt2': (xmax,ymax,zmin) - maximum corner
        'v12': (xmax - xmin, ymax - ymin, zmax - zmin) - x-, y-, z-dimension of bounding box
    """
    xmin = coords[0][0]  ;   xmax = coords[0][0]
    ymin = coords[0][1]  ;   ymax = coords[0][1]
    zmin = coords[0][2]  ;   zmax = coords[0][2]
    for xy in coords[1:]:
        x, y = xy
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    return {'pt1': (xmin,ymin), 'pt2': (xmax,ymax), 
        'v12': (xmax - xmin, ymax - ymin, zmax - zmin)}


# ===========================
# ========== 3 D ============
# ===========================

def cart2cyl(vec3D, default_ang=0):
    """Converts a 3D cartesian vector (x, y, z) to 
    cylindrical coordinates (r, theta, z).
    
    Note that the third coordinate is simply passed through
    (and in fact can be any type and any length - in other words
    it can also handle 2D and nD where n > 1)."""
    x, y, *z = vec3D
    r = (x**2 + y**2)**0.5
    ang = default_ang if x == y == 0 else atan2(y,x)
    return (r, ang, *z)


def cyl2cart(vec3D):
    """Converts a 3D cylindrical vector (r, theta, z) to 
    cartesian coordinates (x, y, z).

    Note that the third coordinate is simply passed through
    (and in fact can be any type and any length - in other words
    it can also handle 2D and nD where n > 1)."""
    r, theta, *z = vec3D
    return (r * cos(theta), r * sin(theta), *z)


def angfix(ang):
    """Keeps an angle in radians within -pi > angle > pi"""
    return ((ang + pi) % (2*pi)) - pi


def cyl_rot3D(v, ang):
    """Rotates 3D cylindrical coordinates (v) by an angle (ang) in radians"""
    r, theta, z = v
    return r, angfix(theta + ang), z


def dist3D(pt1, pt2):
    """Returns distance between two 3D points (as two 3-tuples)"""
    return ((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 + (pt2[2]-pt1[2])**2)**0.5


def mag3D(v3D):
    """Magnitude of the point or vector"""
    return (v3D[0]**2 + v3D[1]**2 + v3D[2]**2)**0.5


def neg3D(v3D):
    """Returns the negative of the 3D vector"""
    return -v3D[0], -v3D[1], -v3D[2]


def unit3D(v, default=None):
    """Returns the unit 3D vector in the same direction 
    as the input 3D vector (as one 3-tuple)"""    
    if mag3D(v) == 0.0 :
        if default is None:
            raise ValueError('Vector should have non-zero magnitude')
        else:
            return default
    else:
        return scale3D(v, 1.0 / mag3D(v))


def add3D(v1,v2):
    """Vector subtraction of v2 from v1"""
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


def sub3D(v1,v2):
    """Vector subtraction of v2 from v1"""
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def scale3D(v,scale):
    """Returns a scaled 3D vector"""
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def dot3D(v1, v2):
    """Calculates the scalar dot product of two 3D vectors, v1 and v2"""
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] 


def cross3D(v1, v2):
    """Calculates the vector cross product of two 3D vectors, v1 and v2"""
    return (v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0])


def sin3D(v1, v2, as_scalar = True):
    """calculates the sine of the angle between two 3D vectors, v1 and v2
    using the magnitude of the cross product of the normalised vectors."""    
    if mag3D(v1) * mag3D(v2) == 0.0 :
        raise ValueError('Both vectors should have non-zero magnitude')
    else:
        scale = 1 / mag3D(v1) / mag3D(v2)
        vec = scale3D(cross3D(v1, v2), scale)
        return mag3D(vec) if as_scalar else vec


def cos_sim3D(v1, v2, default=None):
    """returns the cosine of the angle between two vectors (v1, v2) based on the dot product
    v1 . v2 = |v1|  |v2| cos (<v1,v2>)"""
    if mag3D(v1) * mag3D(v2) == 0.0 :
        if default is None:
            raise ValueError('Both vectors should have non-zero magnitude')
        else:
            return default
    else:
        return dot3D(v1, v2) / mag3D(v1) / mag3D(v2)


def dir_cos3D(v, default=None):
    """returns direction cosines for the vector v
    """
    return tuple([cos_sim3D(v, base, default=default) for base in ((1,0,0), (0,1,0), (0,0,1))])


def dev3D(v1,v2, default=None):
    """Provides the perpendicular distance (deviation) of the end of line v1 from line v2"""
    v1mag = mag3D(v1, 0)
    v2mag = mag3D(v2, 0)
    if v1mag * v2mag == 0:
        if default is None:
            raise ValueError('Both vectors should have non-zero magnitude')
        else:
            return default
    else:
        return mag3D(cross3D(v1,v2)) / v2mag


#def planar_angle(v1, v2, normal=(0,0,1)):
#    """returns the angle of one vector relative to the other in the 
#    plane defined by the normal (default is in the XY plane)
#    NB This algorithm avoids carrying out a coordinate transformation 
#    of both vectors. However, it only works if both vectors are in that 
#    plane to start with. """
#    return atan2(dot3D(normal, sin3D(v1, v2, False)), cos_sim3D(v1, v2))


def rotQ(self, ang, other):
    """Returns the vector rotated by an angle of 'ang' radians about the axis
    vector (other) by means of quaternions"""
    q_w = cos(0.5*ang)
    q_xyz = scale3D(other, sin(0.5*ang))
    t = cross3D(q_xyz, self)
    #print(self, scale3D(t, 2.0 * q_w), cross3D(q_xyz,scale3D(t, 2.0)), sep=' : ')
    #return self + scale3D(t, 2.0 * q_w) + cross3D(q_xyz,scale3D(t, 2.0))
    return add3D(add3D(self, scale3D(t, 2.0 * q_w)), cross3D(q_xyz,scale3D(t, 2.0)))


def rotR(self, ang, other):
    """Returns the vector rotated by an angle of 'ang' radians about the axis
    vector (other) by means of Rodrigues' formula"""
    #print(self, scale3D(self, cos(ang)), scale3D(cross3D(other, self), sin(ang)), 
    #      scale3D(other, (1.0 - cos(ang)) * (dot3D(self, other))), sep=' : ')
    #return scale3D(self, cos(ang)) + scale3D(cross3D(other, self), sin(ang)) + \
    #    scale3D(other, (1.0 - cos(ang)) * (dot3D(self, other)))
    return add3D(add3D(scale3D(self, cos(ang)), scale3D(cross3D(other, self), sin(ang))), 
        scale3D(other, (1.0 - cos(ang)) * (dot3D(self, other))))


def fmt_3x3(dc, fmt='7.4f'):
    """For pretty-printing a 3x3 matrix of unit vectors (3-tuple of 3-tuples)
    NB This is used by test functions, so do not change in isolation"""
    #return ', '.join([f'({x:6.3f}, {y:6.3f}, {z:6.3f})' for x, y, z in dc])
    return ', '.join([f'({x:{fmt}}, {y:{fmt}}, {z:{fmt}})' for x, y, z in dc])


def bbox3D(coords):
    """Runs through x,y,z tuple triplets and extracts the values corresponding to
        'pt1': (xmin,ymin,zmin) - minimum corner
        'pt2': (xmax,ymax,zmin) - maximum corner
        'v12': (xmax - xmin, ymax - ymin, zmax - zmin) - x-, y-, z-dimension of bounding box
    """
    xmin = coords[0][0]  ;   xmax = coords[0][0]
    ymin = coords[0][1]  ;   ymax = coords[0][1]
    zmin = coords[0][2]  ;   zmax = coords[0][2]
    for xyz in coords[1:]:
        x, y, z = xyz
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
        if z < zmin: zmin = z
        if z > zmax: zmax = z
    return {'pt1': (xmin,ymin,zmin), 'pt2': (xmax,ymax,zmax), 
        'v12': (xmax - xmin, ymax - ymin, zmax - zmin)}


def extend_bbox3D(coords, tol = 0.000001, b_end_1 = True, b_end_2 = True):
    """Returns 3D bounding box coordinates extended by some proportion ('tol') in all
    directions of the 3D dimensions of the box (new dimensions are all increased by a 
    factor of (1 + 2 * tol)).
    input:
        bbox - 3D bounding box defined as tuple of tuples ((xmin,ymin,zmin),(xmax,ymax,zmax))
        tol - the ratio of the dimension by which bbox is to be extended (default 0.000001)
        b_end_1 - (True / False) whether to extend end 1 (default True)
        b_end_2 - (True / False) whether to extend end 2 (default True)
    output:
        extended 3D bounding box - defined as tuple of tuples
    """
    bbox_dict = bbox3D(coords)
    end_1 = bbox_dict['pt1']
    end_2 = bbox_dict['pt2']
    dims = bbox_dict['v12']
    delta_dim = scale3D(dims,tol)
    return (sub3D(end_1,delta_dim) if b_end_1 else end_1), (add3D(end_2,delta_dim) if b_end_1 else end_1)


def length_interp_z(z1, z2, z3, l1, l2, l3, l01, l02, l03, n=1.0):
    """Returns the interpolated z-value for a 3D point relative to z-values 
    defined at three points. For n=1, this is equal to a linear interpolation
    - i.e. the assumption that the z-values lie on a plane
    
    The weighting is defined based on the distances between the point 
    of interest (p0) and the points p1, p2 & p3 with z-values z1, z2, & z3):
        l1, l2 & l3 are the lengths of the opposite sides to points p1, p2 & p3
        l01, l02 & l03 are the lengths from the point of interest (p0) to points p1, p2 & p3
        a0 is the area subtended by sides of the triangle p1, p2, p3
        a1 is the area subtended by sides of the sub-triangle p0, p2, p3
        a2 is the area subtended by sides of the sub-triangle p1, p0, p3
        a3 is the area subtended by sides of the sub-triangle p1, p2, p0
    
    The following conclusions may be drawn:
    - p0 is inside the triangle if a0 = a1 + a2 + a3
    - p0 is outside of line 3 (opposite p3) if a0 = a1 + a2 - a3
    - p0 is outside of lines 2 & 3 (near p1) if a0 = a1 - a2 - a3
    """
    # a0 = sides2area(l1, l2, l3) # not used
    a1 = sides2area(l1, l02, l03)
    a2 = sides2area(l2, l03, l01)
    a3 = sides2area(l3, l01, l02)
    sum_a = a1**n + a2**n + a3**n
    # the result (for n = 1) is the same as a linear interpolation
    return (a1**n * z1 + a2**n * z2 + a3**n * z3) / sum_a


# ===========================
# ========== n D ============
# ===========================

def magND(v):
    """Returns magnitude of an nD vector"""
    return sum(vv**2 for vv in v) ** 0.5


def magNDx(v, limit=0):
    """Returns magnitude of an nD vector,
    ignoring items if they are not numeric, with 
    an option to limit length of the tuple to a certain 
    length defined by the `limit` argument"""
    if limit > 0:
        return sum(vv**2  for i, vv in enumerate(v)
        if (isinstance(vv, (int, float)) 
            and i < limit))**0.5
    else:
        return sum(vv**2  for i, vv in enumerate(v)
        if (isinstance(vv, (int, float))))**0.5


def negND(v):
    """Returns negative of an nD vector"""
    return [-vv for vv in v]


def negNDx(v, limit=0):
    """Returns negative of an nD vector,
    ignoring items if they are not numeric, with 
    an option to limit length of the tuple to a certain 
    length defined by the `limit` argument"""
    if limit > 0:
        return [-vv  for i, vv in enumerate(v)
        if (isinstance(vv, (int, float)) 
            and i < limit)]
    else:
        return [-vv  for i, vv in enumerate(v)
        if (isinstance(vv, (int, float)))]


def unitND(v):
    """Returns the unit nD vector in the same direction 
    as the input nD vector (as one 3-tuple)"""    
    if magND(v) == 0.0 :
        raise ValueError('Vector should have non-zero magnitude')
    else:
        return scaleND(v, 1.0 / magND(v))


def addND(v1, v2):
    """Adds two nD vectors together, itemwise"""
    return [vv1 + vv2 for vv1, vv2 in zip(v1, v2)]


def addNDx(v1, v2, limit=0):
    """Adds two nD vectors together, itemwise,
    ignoring items if they are not numeric, with 
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument"""
    if limit > 0:
        return [vv1 + vv2 for i, (vv1, vv2) in enumerate(zip(v1, v2)) 
        if (isinstance(vv1, (int, float)) 
            and isinstance(vv2, (int, float))
            and i < limit)]
    else:
        return [vv1 + vv2 for vv1, vv2 in zip(v1, v2) 
        if (isinstance(vv1, (int, float)) and isinstance(vv2, (int, float)))]


def subND(v1, v2):
    """Subtracts two nD vectors together, itemwise"""
    return [vv1 - vv2 for vv1, vv2 in zip(v1, v2)]


def subNDx(v1, v2, limit=0):
    """Subtracts two nD vectors together, itemwise,
    ignoring items if they are not numeric, with 
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument
    
    e.g.    subNDx((2,4,1,'a'),(7,2,8,'b')) 
        and 
            subNDx((2,4,1,'a'),(7,2,8,'b'), limit=3)
        will both return [-5,2,-7] 
    """
    if limit > 0:
        return [vv1 - vv2 for i, (vv1, vv2) in enumerate(zip(v1, v2)) 
        if (isinstance(vv1, (int, float)) 
            and isinstance(vv2, (int, float))
            and i < limit)]
    else:
        return [vv1 - vv2 for vv1, vv2 in zip(v1, v2) 
        if (isinstance(vv1, (int, float)) and isinstance(vv2, (int, float)))]


def mulND(v1, v2):
    """Returns product of two nD vectors
    (same as itemwise multiplication)"""
    return [vv1 * vv2 for vv1, vv2 in zip(v1, v2)]


def mulNDx(v1, v2, limit=0):
    """Multiplies two nD vectors together, itemwise,
    ignoring items if they are not numeric, with 
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument"""
    if limit > 0:
        return [vv1 * vv2 for i, (vv1, vv2) in enumerate(zip(v1, v2)) 
        if (isinstance(vv1, (int, float)) 
            and isinstance(vv2, (int, float))
            and i < limit)]
    else:
        return [vv1 * vv2 for vv1, vv2 in zip(v1, v2) 
        if (isinstance(vv1, (int, float)) and isinstance(vv2, (int, float)))]


def divND(v1, v2):
    """Returns divisor of two nD vectors
    (same as itemwise multiplication)
    Note that this does not catch cases 
    where an element of the divisor is zero."""
    return [vv1 / vv2 for vv1, vv2 in zip(v1, v2)]


def divNDx(v1, v2, limit=0):
    """Returns itemwise divisor of two nD vectors, 
    ignoring items if one or more are not numeric, with 
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument
    Note that this does not catch cases 
    where an element of the divisor is zero."""
    if limit > 0:
        return [vv1 * vv2 for i, (vv1, vv2) in enumerate(zip(v1, v2)) 
        if (isinstance(vv1, (int, float)) 
            and isinstance(vv2, (int, float))
            and i < limit)]
    else:
        return [vv1 * vv2 for vv1, vv2 in zip(v1, v2) 
        if (isinstance(vv1, (int, float)) and isinstance(vv2, (int, float)))]


def dotND(v1, v2):
    """Returns dot product of two nD vectors
    (same as itemwise multiplication followed by summation)."""
    return sum(vv1 * vv2 for vv1, vv2 in zip(v1, v2))


def dotNDx(v1, v2, limit=0):
    """Returns dot product of two nD vectors, 
    ignoring items if one or more are not numeric, with 
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument
    (same as itemwise multiplication followed by summation).
    Note that this does not catch cases 
    where an element of the divisor is zero."""
    if limit > 0:
        return sum(vv1 * vv2 for i, (vv1, vv2) in enumerate(zip(v1, v2))
        if (isinstance(vv1, (int, float)) 
            and isinstance(vv2, (int, float))
            and i < limit))
    else:
        return sum(vv1 * vv2 for vv1, vv2 in zip(v1, v2)
        if (isinstance(vv1, (int, float)) and isinstance(vv2, (int, float))))


def distND(pt1, pt2):
    """Returns distance between two nD points (as two n-tuples)"""
    return (sum((vv2 - vv1)**2.0 for vv1, vv2 in zip(pt1, pt2)))**0.5


def distNDx(pt1, pt2, limit=0):
    """Returns distance between two nD points (as two n-tuples)
    It ignores items if they are not numeric, and also has
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument"""
    if limit > 0:
        return (sum((vv2 - vv1)**2.0 for i, (vv1, vv2) in enumerate(zip(pt1, pt2))
        if isinstance(vv1, (int, float))
        and isinstance(vv2, (int, float))
        and i < limit))**0.5
    else:
        return (sum((vv2 - vv1)**2.0 for vv1, vv2 in zip(pt1, pt2)
        if isinstance(vv1, (int, float))
        and isinstance(vv2, (int, float))))**0.5


def scaleND(v, s):
    """Scales an nD vector by a factor s."""
    return [s * vv for vv in v]


def scaleNDx(v, s, limit=0):
    """Multiplies two nD vectors together, itemwise,
    ignoring items if they are not numeric, and also 
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument"""
    if limit > 0:
        return [s * vv for i, vv in enumerate(v) 
        if (isinstance(v, (int, float)) 
            and i < limit)]
    else:
        return [s * vv for vv in v 
        if isinstance(v, (int, float))]


def cos_simND(v1, v2, default=None):
    """returns the cosine of the angle between two vectors (v1, v2) based on the dot product
    v1 . v2 = |v1|  |v2| cos (<v1,v2>), with the option of a default
    value in the case of a zero-magnitude input"""
    m1, m2 = magND(v1), magND(v2)
    if m1 * m2 == 0.0 :
        if default is None:
            raise ValueError('Both vectors should have non-zero magnitude')
        else:
            return default
    else:
        return dotND(v1, v2) / m1 / m2


def cos_simNDx(v1, v2, limit=0, default=None):
    """returns the cosine of the angle between two vectors (v1, v2) based on the dot product
    v1 . v2 = |v1|  |v2| cos (<v1,v2>),
    ignoring items if they are not numeric, and also with
    an option to limit length of tuples to a certain 
    length defined by the `limit` argument."""
    m1, m2 = magNDx(v1, limit=limit), magNDx(v2, limit=limit)
    if m1 * m2 == 0.0 :
        if default is None:
            raise ValueError('Both vectors should have non-zero magnitude')
        else:
            return default
    else:
        return dotNDx(v1, v2, limit=limit) / m1 / m2



# ================================
# ==== Matrix Manipulations ======
# ================================

def Madd(M1, M2):
    """Matrix addition (elementwise)"""
    return [[a+b for a, b in zip(c, d)] for c, d in zip(M1, M2)]


def Msub(M1, M2):
    """Matrix subtraction (elementwise)"""
    return [[a-b for a, b in zip(c, d)] for c, d in zip(M1, M2)]


def Mtranspose(M):
    """Matrix transposition"""
    return list(zip(*M))


def Mmult(M1, M2):
    """Matrix multiplication (no checks)"""
    return [[sum(a*b for a, b in zip(c, d)) for c in Mtranspose(M2)] for d in M1]


def MI(n: int):
    """nxn identity matrix"""
    return [[1 if i==j else 0 for i in range(n)] for j in range(n)]


def Minv(matrix):
    """Returns the inverse of a square matrix using Gaussian elimination
    Raises a ValueError if matrix is non-square or singular"""
    n = len(matrix)
    if sum(len(matrix)!=n for m in matrix):
        raise ValueError('Matrix is not square.')
    
    Mm = matrix.copy() # otherwise original will be modified
    Im = MI(n)
    
    for i in range(n):
        k = 1
        s = Mm[i][i]
        while s==0 and i+k < n:
            Mm[i] = addND(Mm[i], Mm[i+k])
            Im[i] = addND(Im[i], Im[i+k])
            k+=1
            s = Mm[i][i]
        if s == 0:
            err_msg = 'Singular Matrix\n' 
            err_msg += f'Processing row {i+1} of {n}:\n' 
            err_msg +=  repr(Mm)
            raise ValueError(err_msg)
        
        Mm[i] = scaleND(Mm[i], 1 / s)
        Im[i] = scaleND(Im[i], 1 / s)
        for j in range(n):
            if j!=i and Mm[j][i] != 0:
                s = Mm[j][i]
                Mm[j] = subND(Mm[j], scaleND(Mm[i], s))
                Im[j] = subND(Im[j], scaleND(Im[i], s))
    return Im #, Mm


def Mdet(matrix):
    """Returns the determinant of a square matrix using Gaussian elimination
    
    Determinant is calculated by Gaussian elimination method
    A ValueError is raised if matrix is non-square or singular
    
    Args:
        matrix:
    
    """
    n = len(matrix)
    if sum(len(m)!=n for m in matrix):
        raise ValueError('Matrix is not square.')
    
    M = matrix.copy() # otherwise original will be modified

    det_sign = 1
    
    for i in range(n):
        k=1        
        s = M[i][i]
        # Switch rows if diagonal is zero
        while s==0 and i+k < n:
            M[i], M[i+k] = M[i+k], M[i]
            det_sign *= -1
            k+=1
            s = M[i][i]
        if s == 0:
            err_msg = 'Singular Matrix\n' 
            err_msg += f'Processing row {i+1} of {n}:\n' 
            err_msg +=  repr(M)
            raise ValueError(err_msg)
        
        # eliminate value in i-th column in remainimg rows
        for j in range(i+1, n):
            if M[j][i] != 0:
                s = M[j][i] / M[i][i]
                M[j] = subND(M[j], scaleND(M[i], s))
    
    #print(M)
    return det_sign * reduce(mul, [M[i][i] for i in range(n)], 1)


def kron():
    """
    TO DO - Kronecker product"""
    pass


def sides2area(l1, l2, l3):
    """Heron's formula - area of triangle calculated from lengths of 3 sides"""
    s = 0.5 * (l1 + l2 + l3)
    return (s * (s - l1) * (s - l2) * (s - l3))**0.5


def main():
    #import numpy as np
    print("Hello World!")
    A = [[1, 2], [4, -2]]
    B = [[4, -1], [-3, 5]]
    C = [[0, 3], [1, 5]]
    print('A', A)
    print('B', B)
    print('C', C)
    print('A+B', Madd(A,B))
    print('A*B', Mmult(A,B))
    print('det(A)', Mdet(A))
    
    A = [[1, 3, 2], [4, -1, 2], [2, 4, -1]]
    B = [[1, 3, 2], [4, -1, 2], [-3, 1, 5]]
    C = [[0, 3, 2], [0, 0, 2], [-3, 1, 5]]
    D = [[0, 0, 2], [0, 3, 0], [-5, 0, 0]]
    A[0], A[1] = A[1], A[0]
    print('A', A)
    print('B', B)
    print('C', C)
    print('A+B', Madd(A,B))
    print('A*B', Mmult(A,B))
    print('det(A)', Mdet(A))
    print('det(B)', Mdet(B))
    print('det(C)', Mdet(C))
    print('det(D)', Mdet(D))
    #print('C', C)
    print('inv(A)', Minv(A))
    print('inv(D)', Minv(D))
    
    #print('det(A)', np.linalg.det(np.array(A)))
    #print('det(B)', np.linalg.det(np.array(B)))
    #print('det(C)', np.linalg.det(np.array(C)))
    #print('det(D)', np.linalg.det(np.array(D)))
    #print('inv(A)', np.linalg.inv(np.array(A)))
    #print('inv(D)', np.linalg.inv(np.array(D)))
    

if __name__ == "__main__":
    main()
