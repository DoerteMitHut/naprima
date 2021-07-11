import numpy as np
#from scipy.linalg import expm,logm,pinv
#import modern_robotics as mr
#import warnings
#import time




def listify(a):
    '''returns the a list of the given numpy array a'''
    return list(np.array(a).flatten())

def normalize(v,min_norm = 10e-10):
    '''returns the normalized vector v unless its euclidian norm is < min_norm which defaults to 10e-10. Returns the appropriately shaped zero vector if ||v||< min_norm '''
    n = np.linalg.norm(v)
    if n > min_norm:
        return v/np.linalg.norm(v)
    else:
        return np.zeros(np.array(v).shape)

def homogenous(p,position = False):
    """transforms a 3- or 4-element list or array into a 4-element one by setting a fourth element to 0 by default and 1 if position is True."""
    if len(p) == 4:
        p[3] = int(position)
    elif len(p) == 3:
        p = np.append(p,int(position))
    else:
        raise ValueError("cannot convert array of shape {} to homogenous vector.".format(p.shape))
    return p

def cartesian(p):
    """returns the first 3 elements of a given list or array as a (1,3) row vector"""
    return np.array(p[0:3]).reshape((1,3))

def row(v):
    """reshapes a list or array to be a row vector"""
    v = np.array(v)
    return v.reshape((1,max(v.shape)))

def column(v):
    """reshapes a list or array to be a column vector"""
    v = np.array(v)
    return v.reshape(max(v.shape),1)

def adjoint_repr(T):
    '''returns the adjoint representation of the transformation given by the SE(3) matrix T.'''
    R = T[0:-1,0:-1]
    p = T[0:-1,-1].reshape((3,1))
    tr = np.vstack([np.hstack([R,np.zeros((3,3))]),
        np.hstack([skew_so3(p)@R,R])])
    return tr

#//////////////////////////////////////////////////////////////////////////////
#//////////// CODE PARTIALLY MODIFIED FROM modern_robotics PACKAGE ////////////
#//////////////////////////////////////////////////////////////////////////////
'''
***************************************************************************
Modern Robotics: Mechanics, Planning, and Control.
Code Library
***************************************************************************
Author: Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes,
Email: huanweng@u.northwestern.edu
Date: January 2019
***************************************************************************
Language: Python
Also available in: MATLAB, Mathematica
Required library: numpy
Optional library: matplotlib
***************************************************************************
'''

def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix

    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs

    Example Input:
        R = np.array([[1, 0,  0],
          [0, 0, -1],
          [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
      [0, 0, -1, 2],
      [0, 1,  0, 5],
      [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector

    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat

    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
   [ 3,  0, -1, 5],
   [-2,  1,  0, 6],
   [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
     [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
          [0, 0, -1, 0],
          [0, 1,  0, 3],
          [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
       [0, 0, -1],
       [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
   [ 3,  0, -1],
   [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3

    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2, 4],
        [ 3,  0, -1, 5],
        [-2,  1,  0, 6],
        [ 0,  0,  0, 0]])
    """
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
        np.zeros((1, 4))]

def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
      [ 3,  0, -1],
      [-2,  1,  0]])
    """
    omg = listify(omg)
    return np.array([[0,      -omg[2],  omg[1]],
         [omg[2],       0, -omg[0]],
         [-omg[1], omg[0],       0]])

#///////////////////////////////////////////////////////
#///////////END OF modern_robotics FUNCTIONS ///////////
#///////////////////////////////////////////////////////
#/////// CONSTANTS /////
class Stances:
    ''' A collection of arm configurations for 7R kinematic chains'''
    home =[0,0,0,0,0,0,0]
    angled_right_arm = [1.2462, 1.4661, 1.5708, -1.8431, -1.7802, 1.2587e-06, 0.2374]

class Robots:
    '''A Collection of premade kinematic chains'''
    CAREN ={"joint_rotation_axes": np.array(  [[   0,  0,  1],
[   0,  -1, 0],
[   0,  0,  1],
[   0,  1,  0],
[   0,  0,  1],
[   0,  -1, 0],
[   0,  0,  1]]),
"joint_home_positions": np.array([column([0,0,l]) for l in [0,0.31,0.51,0.71,0.905,1.1,1.35]]),
"tool_frame_home": RpToTrans(np.eye(3),column([0,0,1.35])),
"joint_angles_home": None,
"all_joints_KC": True}

def skew_so3(omega):
    """produces a 3x3 skew symmetric matrix m such that for a 3-vector v m*v.T = omega X v"""
    omega = np.array(omega).flatten()
    return np.array([[0         ,   -omega[2]   ,   omega[1]    ],
        [omega[2]   ,   0           ,   -omega[0]   ],
        [-omega[1]  ,   omega[0]    ,   0           ]])
def skew_se3(S):
    """produces a 4x4 skew symmetric matrix from a 6-vector screw axis S to be used in the matrix exponential"""
    S = S.flatten()
    return np.vstack([np.hstack([skew_so3(S[0:3]),S[3:].reshape((3,1))]),np.array([0,0,0,0])])

# def getScrewAxis(n):
#     """produces a 6-vector screw axis for the nth joint of the CAREN arm"""
#     return(np.vstack([OMEGAS[n].reshape((3,1)),np.cross(-OMEGAS[n],np.array([0,0,L[n]])).reshape((3,1))]))

# def FK(thetas):
#     """produces the homogenous spacial coordinates of the end-effector of the CAREN arm"""
#     E = np.eye(4)
#     for i in range(7):
#         E = E @ expm(skew_se3(getScrewAxis(i))*thetas[i])
#     fk = E @ M @ np.array([[0,0,0,1]]).T
#     return fk

# def FK_twist(thetas):
#     """produces the twist to produce transform a point from the tool frame to the space frame"""
#     E = np.eye(4)
#     for i in range(7):
#         E = E @ expm(skew_se3(getScrewAxis(i))*thetas[i])
#     fk = mr.se3ToVec(mr.MatrixLog6(E @ M))
#     return fk

# def FK_se3(thetas):
#     E = np.eye(4)
#     for i in range(7):
#         E = E @ expm(skew_se3(getScrewAxis(i))*thetas[i])
#     fk = mr.MatrixLog6(E @ M)
#     return fk

# def FK_SE3(thetas):
#     E = np.eye(4)
#     for i in range(7):
#         E = E @ expm(skew_se3(getScrewAxis(i))*thetas[i])
#     fk = E @ M
#     return fk

# def getJacobian(screwAxes, thetas):
#     E = np.eye(4)
#     Js = []
#     for i in range(7):
#         Js.append(adjoint_repr(E)@screwAxes[i])
#         E = E @ expm(skew_se3(getScrewAxis(i))*thetas[i])
#     return np.array(Js)[:,:,0].T

# def getThetas():
#     thetas = []
#     for sensor in armMotorPositionSensors:
#         thetas.append(sensor.getValue())
#     return thetas

# def setMotors(theta_dot):
#     if np.max(np.abs(theta_dot)) > 1.95:
#         theta_dot = theta_dot*(1.95/np.max(np.abs(theta_dot)))
#     for i in range(7):
#         armMotors[i].setPosition(float('inf'))
#         #armMotors[i].setPosition(thetas[i])
#         #armMotors[i].setPosition(np.random.rand()*2*2.0944-2.0944)
#         armMotors[i].setVelocity(float(theta_dot[i]))

# def moveTo(target):
#     lrot = get_delta_omega_mat(target,FK_SE3(getThetas()))
#     p = FK(getThetas())
#     d = (target.flatten()[0:3]-(p.flatten()[0:3]))
#     return getTransform(lrot*0.1,d)

# def getRotation(axis,theta):
#     if axis == 'x':
#         return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
#     elif axis == 'y':
#         return np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
#     elif axis == 'z':
#         return np.array([[np.cos(theta),-np.sin(theta),0],[0,0,1],[np.sin(theta),np.cos(theta),0]])
#     else:
#         return np.eye(3)

# def getTransform(R,p):
#     return(np.vstack([np.hstack([R.reshape((3,3)),p.reshape(3,1)]),np.array([0,0,0,0])]))

# def webots2caren(wbCoords,homogenous=True):
#     if wbCoords.size == 3:
#         toRet = np.array([wbCoords]).reshape([3,1])-(ORIGIN_OFFSET[0:3])
#     else:
#         raise Exception("wrong shape %d of Webots Coordinates to be transformed. Should be 3.")
#     if homogenous:
#         toRet = np.append(toRet,1)
#     return toRet

# def get_delta_omega(target,origin):
#     target = target[0:3].reshape([3,1])
#     print("target at\n%s\n"%(target))
#     o_r = origin[0:3,0:3]
#     print("EEf rotated like \n%s\n"%(o_r))
#     o_t = origin[0:3,3].reshape([3,1])
#     print("eef position at \n%s\n"%(o_t))
#     z_bar = target-o_t
#     z_bar[1,0] = 0
#     z_bar = z_bar/np.linalg.norm(z_bar)
#     print("z_bar is \n%s\n"%(z_bar))
#     y_bar = np.array([[0,1,0]]).T
#     print("y_bar is \n%s\n"%(y_bar))
#     x_bar = np.cross(y_bar.flatten(),z_bar.flatten()).reshape([3,1])
#     x_bar = x_bar/np.linalg.norm(x_bar)
#     print("x_bar is \n%s\n"%(x_bar))
#     R2 = np.array([x_bar,y_bar,z_bar]).reshape([3,3])
#     print("R2 is \n%s\nhas det %s"%(R2,np.linalg.det(R2)))
#     delta_rot = logm(R2 @ o_r.T).real
#     R = R2 @ o_r.T
#     theta = np.arccos((np.trace(R)-1)/2)
#     n = 1/(2*np.sin(theta))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]]).reshape([1,3])
#     print("R_dot is \n%s\n"%(delta_rot))
#     print("whoop:",n*theta)
#     return n*theta

# def get_delta_omega_mat(target,origin):
#     target = target[0:3].reshape([3,1])
#     print("target at\n%s\n"%(target))
#     o_r = origin[0:3,0:3]
#     print("EEf rotated like \n%s\n"%(o_r))
#     o_t = origin[0:3,3].reshape([3,1])
#     print("eef position at \n%s\n"%(o_t))
#     z_bar = target-o_t
#     z_bar[1,0] = 0
#     z_bar = z_bar/np.linalg.norm(z_bar)
#     print("z_bar is \n%s\n"%(z_bar))
#     y_bar = np.array([[0,1,0]]).T
#     print("y_bar is \n%s\n"%(y_bar))
#     x_bar = np.cross(y_bar.flatten(),z_bar.flatten()).reshape([3,1])
#     x_bar = x_bar/np.linalg.norm(x_bar)
#     print("x_bar is \n%s\n"%(x_bar))
#     R2 = np.array([x_bar,y_bar,z_bar]).reshape([3,3])
#     print("R2 is \n%s\nhas det %s"%(R2,np.linalg.det(R2)))
#     delta_rot = logm(R2 @ o_r.T).real
#     R = R2 @ o_r.T
#     theta = np.arccos((np.trace(R)-1)/2)
#     n = 1/(2*np.sin(theta))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]]).reshape([1,3])

#     return expm(skew_so3(n*theta))

# def get_inverse_jacobian(thetas):
#     Js = getJacobian([getScrewAxis(i) for i in range(7)], thetas)
#     Js_i = pinv(Js)
#     return Js_i







    