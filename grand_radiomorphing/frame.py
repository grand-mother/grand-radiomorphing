''' Collection of functions used for local frame transforms for pulse shape computations.
'''
import numpy


def get_rotation(zen, az, phigeo, bfieldangle):
    """Utility function for getting the rotation matrix between frames

    Arguments:
    ----------
        zen: float
            zenith angles of primary in radian
        az: float
            azmimuth angle in radian
        phigeo, bfieldangle: floats
            magnetic field angles in radian
    Returns: 
    ---------
        Rotation matrix: numpy.array((v, vxB, vxvxB))
    """
    s = numpy.sin(bfieldangle)
    B = numpy.array([numpy.cos(phigeo) * s, numpy.sin(phigeo) * s,
                     numpy.cos(bfieldangle)])

    s = numpy.sin(zen)
    v = numpy.array([numpy.cos(az) * s, numpy.sin(az) * s, numpy.cos(zen)])

    vxB = numpy.cross(v, B)
    vxB /= numpy.linalg.norm(vxB)
    vxvxB = numpy.cross(v, vxB)
    vxvxB /= numpy.linalg.norm(vxvxB)

    return numpy.array((v, vxB, vxvxB))


def UVWGetter(cx, cy, cz, zen, az, phigeo, bfieldangle):
    """Closure for getting coordinates in the shower frame.

    Arguments:
    ----------
        cx,cy,cz: floats
            positions vector, in meters
        zen: float
            zenith angles of primary in radian
        az: float
            azmimuth angle in radian
        phigeo, bfieldangle: floats
            magnetic field angles in radian
    Returns: 
    ---------
        Rotation matrix: numpy array
    """
    R = get_rotation(zen, az, phigeo, bfieldangle)
    origin = numpy.array((cx, cy, cz))

    def GetUVW(pos):
        return numpy.dot(R, pos - origin)
    return GetUVW


def XYZGetter(cx, cy, cz, zen, az, phigeo, bfieldangle):
    """Closure for getting back to the main frame

    Arguments:
    ----------
        cx,cy,cz: floats
            positions vector, in meters
        zen: float
            zenith angles of primary in radian
        az: float
            azmimuth angle in radian
        phigeo, bfieldangle: floats
            magnetic field angles in radian
    Returns: 
    ---------
        Rotation matrix: numpy array
    """
    Rt = get_rotation(zen, az, phigeo, bfieldangle).T
    origin = numpy.array((cx, cy, cz))

    def GetXYZ(pos):
        return numpy.dot(Rt, pos) + origin
    return GetXYZ
