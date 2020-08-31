import numpy as np
from scipy.linalg import expm
import math


def e2h(x):
    '''convert Euclidean coordinates column vector to homogeneous coordinates'''

    return np.vstack( (x, np.ones(shape=(1,x.shape[1]))) )


def h2e(x):
    '''convert homogeneous coordinates column vector to Euclidean coordinates'''

    return x[0:-1] @ np.diag(1 / x[-1])


class Camera:
    '''Class to represent simple projective camera.
     
      Methods:
        project() return image-plane projection of world points
        visjac() return visual Jacobian
        get_C() return camera matrix
    '''

    def __init__(self, f=8*1e-3, rho=10*1e-6, pp=(500, 500), T=None):
        '''Create an instance of a Camera class.
         
           Optional parameters include:
           `f`    focal length (metres)
           `rho`  pixel size (side length in metres, assumed square)
           `pp`   principal point (pixels)
           `T`    a 4x4 SE3 transform representing the pose of the camera
        '''
        self.f = f
        self.rho = rho
        self.pp = pp

        if T is None:
            self.T = np.identity(4)
        else:
            self.T = T

        self.C = self.get_C(self.T)

    def project(self, P, T=None):
        '''Project world points to image-plane points.

           World points are given as columns of a 3xN matrix.

           .project(P)        return 2xN matrix of 2D image-plane points
           .project(P, T=SE3) as above, but for given camera pose
        '''

        if T is None:
            C = self.C
        else:
            C = self.get_C(T)
        return h2e( C @ e2h(P) )
    
    def get_C(self, T=None):
        '''Return camera matrix
         
           .get_C() is the 3x4 camera matrix for this camera
           .get_C(T=SE3) as above but for the camera at pose given by T
        '''
        K = np.array([  [self.f/self.rho, 0,               self.pp[0]], 
                        [0,               self.f/self.rho, self.pp[1]], 
                        [0,               0,               1]])
        P0 = np.array([ [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
        
        if T is None:
            C = K @ P0 @ np.linalg.inv(self.T)
        else:
            C = K @ P0 @ np.linalg.inv(T)
        
        return C

    def visjac(self, uv, Z):
        '''Image Jacobian (interaction matrix)
        
           Returns a 2Nx6 matrix of stacked Jacobians, one per image-plane point.
           uv is a 2xN matrix of image plane points
           Z  is the depth of the corresponding world points. Can be scalar, same distance to every
           point, or a vector or list of length N.
           
           References:
           * A tutorial on Visual Servo Control", Hutchinson, Hager & Corke, 
             IEEE Trans. R&A, Vol 12(5), Oct, 1996, pp 651-670.
           * Robotics, Vision & Control, Corke, Springer 2017, Chap 15.
        '''

        if numcols(uv) > 1:
            if np.isscalar(Z):
                Z = [Z] * numcols(uv)
            elif np.prod(Z.size) != numcols(uv):
                raise ValueError('Z must be a scalar or have same number of columns as uv')
        elif numcols(uv) == 1:
            Z = [Z]

        L = np.empty((0, 6))  # empty matrix

        for i in range(numcols(uv)):  # iterate over each column (point)

            p = uv[:,i]
            z = Z[i]
            # convert to normalized image-plane coordinates
            x = (p[0] - self.pp[0]) * self.rho / self.f
            y = (p[1] - self.pp[1]) * self.rho / self.f

            # 2x6 Jacobian for this point
            Lp = np.diag([self.f / self.rho]*2) @ np.array(
                [ [-1/z,  0,     x/z, x * y,      -(1 + x**2), y],
                  [ 0,   -1/z,   y/z, (1 + y**2), -x*y,       -x] ])

            # stack them vertically
            L = np.vstack([L, Lp])

        return L


def colvec(v):
    return np.array(v).reshape((len(v), 1))


def transl(x=None, y=None, z=None):
    '''Create an SE3 translation matrix.
       transl(1, 2, 3)
       transl(x=1, y=2, z=3)
       transl([1, 2, 3])
    '''

    if type(x) is list:
        if len(x) == 3:
            temp = np.array([[x[0]], [x[1]], [x[2]]])
            temp = np.concatenate((np.eye(3), temp), axis=1)
            return np.concatenate((temp, np.array([[0, 0, 0, 1]])), axis=0)
    else:
        raise AttributeError("Invalid arguments")


def trexp(se3):
    return expm( skewa(se3) )


def skew(v):
    return np.array( [
            [  0,    -v[2],  v[1] ],
            [  v[2],  0,    -v[0] ],
            [ -v[1],  v[0],  0]
        ])

def skewa(v):
    return np.vstack( (
                       np.hstack( (skew(v[3:6]), colvec(v[0:3])) ),
                       [0, 0, 0, 0]
                       ))


def numcols(A):
    '''docstring
    '''

    return A.shape[1]


def numrows(A):
    '''docstring
    '''

    return A.shape[0]


def colvec(v):
    '''Create a column vector from a 1D list'''

    return np.array(v).reshape((len(v), 1))


def transl(x=None, y=None, z=None):
    '''Create an SE3 translation matrix.
       transl(1, 2, 3)
       transl(x=1, y=2, z=3)
       transl([1, 2, 3])
    '''

    if type(x) is list:
        if len(x) == 3:
            temp = np.array([[x[0]], [x[1]], [x[2]]])
            temp = np.concatenate((np.eye(3), temp), axis=1)
            return np.concatenate((temp, np.array([[0, 0, 0, 1]])), axis=0)
    else:
        raise AttributeError("Invalid arguments")


# ---------------------------------------------------------------------------------------#
def rotx(theta, *, unit="rad"):
    """
    ROTX gives rotation about X axis

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: rotation matrix

    rotx(THETA) is an SO(3) rotation matrix (3x3) representing a rotation
    of THETA radians about the x-axis
    rotx(THETA, "deg") as above but THETA is in degrees

    THETA is a scalar (float or int), a 1D sequence (list, tuple or ndarray) or any
    other kind of iterable object that returns a sequence of scalars.
    """
    if unit == "deg":
        conv = math.pi / 180
    else:
        conv = 1

    theta *= conv
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.array([
        [1, 0, 0],
        [0, ct, -st],
        [0, st, ct]
    ])
    return mat.round(15)


# ---------------------------------------------------------------------------------------#
def roty(theta, unit="rad"):
    """
    ROTY Rotation about Y axis

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: rotation matrix

    roty(THETA) is an SO(3) rotation matrix (3x3) representing a rotation
    of THETA radians about the y-axis
    roty(THETA, "deg") as above but THETA is in degrees
    """
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    return mat.round(15)


# ---------------------------------------------------------------------------------------#
def rotz(theta, unit="rad"):
    """
    ROTZ Rotation about Z axis

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: rotation matrix

    rotz(THETA) is an SO(3) rotation matrix (3x3) representing a rotation
    of THETA radians about the z-axis
    rotz(THETA, "deg") as above but THETA is in degrees
    """
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    return mat.round(15)


# ---------------------------------------------------------------------------------------#
def trotx(theta, unit="rad", xyz=[0, 0, 0]):
    """
    TROTX Rotation about X axis

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param xyz: the xyz translation, if blank defaults to [0,0,0]
    :return: homogeneous transform matrix

    trotx(THETA) is a homogeneous transformation (4x4) representing a rotation
    of THETA radians about the x-axis.
    trotx(THETA, 'deg') as above but THETA is in degrees
    trotx(THETA, 'rad', [x,y,z]) as above with translation of [x,y,z]
    """
    if unit == "deg":
        theta = theta * math.pi / 180
        unit = "rad"
    tm = rotx(theta, unit=unit)
    tm = np.r_[tm, np.zeros((1, 3))]
    mat = np.c_[tm, np.array([[xyz[0]], [xyz[1]], [xyz[2]], [1]])]
    mat = mat.round(15)
    return mat


# ---------------------------------------------------------------------------------------#
def troty(theta, unit="rad", xyz=[0, 0, 0]):
    """
    TROTY Rotation about Y axis

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param xyz: the xyz translation, if blank defaults to [0,0,0]
    :return: homogeneous transform matrix

    troty(THETA) is a homogeneous transformation (4x4) representing a rotation
    of THETA radians about the y-axis.
    troty(THETA, 'deg') as above but THETA is in degrees
    troty(THETA, 'rad', [x,y,z]) as above with translation of [x,y,z]
    """
    if unit == "deg":
        theta = theta * math.pi / 180
        unit = "rad"
    tm = roty(theta, unit=unit)
    tm = np.r_[tm, np.zeros((1, 3))]
    mat = np.c_[tm, np.array([[xyz[0]], [xyz[1]], [xyz[2]], [1]])]
    mat = (mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def trotz(theta, unit="rad", xyz=[0, 0, 0]):
    """
    TROTZ Rotation about Z axis

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param xyz: the xyz translation, if blank defaults to [0,0,0]
    :return: homogeneous transform matrix

    trotz(THETA) is a homogeneous transformation (4x4) representing a rotation
    of THETA radians about the z-axis.
    trotz(THETA, 'deg') as above but THETA is in degrees
    trotz(THETA, 'rad', [x,y,z]) as above with translation of [x,y,z]
    """
    if unit == "deg":
        theta = theta * math.pi / 180
        unit = "rad"
    tm = rotz(theta, unit=unit)
    tm = np.r_[tm, np.zeros((1, 3))]
    mat = np.c_[tm, np.array([[xyz[0]], [xyz[1]], [xyz[2]], [1]])]
    mat = mat.round(15)
    return mat