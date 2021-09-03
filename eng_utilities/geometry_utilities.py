""""""


from math import pi, sin, cos, tan, atan, atan2, asin, acos, exp, log, log10


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
    return (v2D[0]**2 + v2D[1]**2)**0.5


def ang2D(pt):
    return atan2(pt[1],pt[0])


def sub2D(v1,v2):
    """Vector subtraction of v2 from v1"""
    return (v1[0] - v2[0], v1[1] - v2[1])


def outer2D(v1, v2):
    """Calculates the magnitude of the outer product of two 2D vectors, v1 and v2"""
    return v1[0]*v2[1] - v1[1]*v2[0]


def inner2D(v1, v2):
    """Calculates the inner product of two 2D vectors, v1 and v2"""
    return v1[0]*v2[0] + v1[1]*v2[1]


def cos_sim2D(v1, v2):
    """returns the cosine of the angle between two vectors (v1, v2) based on the dot product
    v1 . v2 = |v1|  |v2| cos (<v1,v2>)
    A result of 1 means they are parallel, -1 that they are anti-parallel,
    and 0 that they are perpendicular."""
    if mag2D(v1) * mag2D(v2) == 0.0 :
        #raise ValueError('Both vectors should have non-zero magnitude')
        return 1
    else:
        return inner2D(v1, v2) / mag2D(v1) / mag2D(v2)


def rotate2D(pt, ang):
    """Rotates a point (x, y) about the origin by an angle in radians"""
    return mag2D(pt) * cos(ang + ang2D(pt)), mag2D(pt) * sin(ang + ang2D(pt))


# ============================================
# ============= 3 D ===========

def dist3D(pt1, pt2):
    """Returns distance between two 3D points (as two 3-tuples)"""
    return ((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 + (pt2[2]-pt1[2])**2)**0.5


def mag3D(v3D):
    """Magnitude of the point or vector"""
    return (v3D[0]**2 + v3D[1]**2 + v3D[2]**2)**0.5


def unit3D(v):
    """Returns the unit 3D vector in the same direction 
    as the input 3D vector (as one 3-tuple)"""    
    if mag3D(v) == 0.0 :
        raise ValueError('Vector should have non-zero magnitude')
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


def sin3D(v1, v2):
    """calculates the sine of the angle between two 3D vectors, v1 and v2
    using the magnitude of the cross product of the normalised vectors."""    
    if mag3D(v1) * mag3D(v2) == 0.0 :
        raise ValueError('Both vectors should have non-zero magnitude')
    else:
        return mag3D(cross3D(v1, v2)) / mag3D(v1) / mag3D(v2)


def cos_sim3D(v1, v2):
    """returns the cosine of the angle between two vectors (v1, v2) based on the dot product
    v1 . v2 = |v1|  |v2| cos (<v1,v2>)"""
    if mag3D(v1) * mag3D(v2) == 0.0 :
        raise ValueError('Both vectors should have non-zero magnitude')
    else:
        return dot3D(v1, v2) / mag3D(v1) / mag3D(v2)


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


def magND(v):
    return sum(vv*2 for vv in v) ** 0.5


def unitND(v):
    s = magND(v)
    return [s * vv for vv in v]


def addND(v1, v2):
    return [vv1 + vv2 for vv1, vv2 in zip(v1, v2)]


def subND(v1, v2):
    return [vv1 - vv2 for vv1, vv2 in zip(v1, v2)]


def dotND(v1, v2):
    return sum(vv1 * vv2 for vv1, vv2 in zip(v1, v2))


def scaleND(v, s):
    return [s * vv for vv in v]


def Madd(M1, M2):
    return [[a+b for a, b in zip(c, d)] for c, d in zip(M1, M2)]


def Mtranspose(M):
    return list(zip(*M))


def Mmult(M1, M2):
    return [[sum(a*b for a, b in zip(c, d)) for c in Mtranspose(M2)] for d in M1]


def MI(n):
    return [[1 if i==j else 0 for i in range(n)] for j in range(n)]


def Minv(M):
    """Check that diagonals are not zero..."""
    n = len(M)
    if sum(len(M)!=n for m in M):
        raise ValueError('Matrix is not square.')
    
    Mm = M.copy()
    Im = MI(n)
    k=1
    
    for i in range(n):
        s = Mm[i][i]
        while s==0 and i+k < n:
            Mm[i] = addND(Mm[i], Mm[i+k])
            Im[i] = addND(Im[i], Im[i+k])
            k+=1
            s = Mm[i][i]
        if s == 0:
            raise ValueError('Singular Matrix')
        Mm[i] = scaleND(Mm[i], 1 / s)
        Im[i] = scaleND(Im[i], 1 / s)
        for j in range(n):
            if j!=i and Mm[j][i] != 0:
                s = Mm[j][i]
                Mm[j] = subND(Mm[j], scaleND(Mm[i], s))
                Im[j] = subND(Im[j], scaleND(Im[i], s))
    return Im #, Mm
