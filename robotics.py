from naprima.utils import *
import numpy as np
from scipy.linalg import expm, pinv
class KinematicChain:
    def __init__(   self,joint_rotation_axes    = None,
                    joint_home_positions        = None,
                    tool_frame_home             = None,
                    joint_angles_home           = None,
                    all_joints_KC               = False):
        self.joint_rotation_axes = joint_rotation_axes
        self.joint_home_positions = joint_home_positions
        self.tool_frame_home = tool_frame_home
        self.n_joints = len(self.joint_home_positions)
        if joint_angles_home:
            self.joint_angles_home = joint_angles_home
        else:
            self.joint_angles_home = [0 for i in range(self.n_joints)]

        self.joint_KCs = None
        if all_joints_KC:
            self.joint_KCs = [KinematicChain(self.joint_rotation_axes[0:i],
                joint_home_positions[0:i],
                RpToTrans(np.eye(3),joint_home_positions[i]),
                joint_angles_home=self.joint_angles_home[0:i]) for i in range(self.n_joints)]
    def _get_link_lengths(self,thetas):
        return [np.linalg.norm(self.joint_KCs[i].get_fk_position(thetas)-self.joint_KCs[i-1].get_fk_position(thetas)) for i in range(1,len(self.joint_KCs))]

    def get_screw_axis(self,n):
        """produces a 6-vector screw axis for the nth joint of the CAREN arm"""
        return(np.vstack([column(self.joint_rotation_axes[n]),column(np.cross(-row(self.joint_rotation_axes[n]),row(self.joint_home_positions[n])))]))

    def get_fk_position(self,thetas):
        """produces a 3-vector describing the end effector position w.r.t the base frame"""
        return column(self.get_fk_SE3(thetas)[0:3,3])

    def get_fk_orientation(self,thetas):
        """produces the 3x3 SO(3) matrix representation of the end effector orientation w.r.t. the base frame"""
        return self.get_fk_SE3(thetas)[0:3,0:3]

    def get_fk_twist_vector(self,thetas):
        """produces the twist vector that transforms the base frame into the tool frame"""
        return se3ToVec(self.get_fk_se3(thetas))

    def get_fk_se3(self,thetas):
        """produces the matrix logarithm of the SE(3) matrix that describes the transformation of the base frame into the tool frame"""
        return scipy.linalg.logm(self.get_fk_SE3(thetas))

    def get_fk_SE3(self,thetas):
        """produces the SE(3) matrix that describes the transformation of the base frame into the tool frame"""
        thetas = listify(thetas)
        E = np.eye(4)
        for i in range(len(self.joint_angles_home)):
            E = E @ expm(skew_se3(self.get_screw_axis(i))*thetas[i])
        fk = E @ self.tool_frame_home
        return fk

    def get_jacobian(self, thetas):
        thetas = listify(thetas)
        E = np.eye(4)
        Js = []
        for i in range(len(self.joint_angles_home)):
            Js.append(adjoint_repr(E) @ self.get_screw_axis(i))
            E = E @ expm(skew_se3(self.get_screw_axis(i))*thetas[i])
        return np.array(Js)[:,:,0].T

    def get_inverse_jacobian(self,thetas):
        Js = self.get_jacobian(thetas)
        Js_i = pinv(Js)
        return Js_i


class Kinematic1DGripper:
    '''A kinematic model for a 1D-two-fingered gripper that resemebles the SCHUNK Dextrous Hand 2.0 where measurements are concerned.'''
    def __init__(self):
        self.proximal_phalanx = 0.0865
        self.distal_phalanx = 0.0685
        self.base_joint_distance = 0.057158
        self.distal_joint_angle = np.pi/4
        self.angle_offset = np.arctan(self.distal_phalanx*np.sin(self.distal_joint_angle)/(self.proximal_phalanx+self.distal_phalanx*np.cos(self.distal_joint_angle))) 
        self.r_tip = self.distal_phalanx*np.sin(self.distal_joint_angle)/(np.sin(self.angle_offset))
        print(f"gripper set up with\n\t angle offset = {self.angle_offset:.3f} rad\n\t tip radius = {self.r_tip*100:.2f} cm")

    def get_fk_shape(self,thetas):
        '''returns the aperture of the gripper for a given joint angle'''
        theta = thetas.flatten()[0]
        return np.array([self.base_joint_distance-2*self.r_tip*np.sin(theta+self.angle_offset)])

    def get_fk_deformation(self,theta_dots):
        '''returns the rate of change of the aperture for a given joint velocity'''
        theta_dot = theta_dots.flatten()[0]
        return np.array([-2*self.r_tip*np.cos(theta+self.angle_offset)*theta_dot])

    def get_ik_shape(self,gammas):
        '''returns a joint angle that realizes a given aperture'''
        aperture = gammas.flatten()[0]
        return np.array([np.arcsin((aperture-self.base_joint_distance)/(-2*self.r_tip))-self.angle_offset])
    
    def get_ik_deformation(self,thetas,gamma_dots):
        '''returns a joint velocity to realize a given rate of aperture change'''
        theta = thetas.flatten()[0]
        aperture_dot = gamma_dots.flatten()[0]
        return np.array([-aperture_dot/(2*np.cos(theta+self.angle_offset)*self.r_tip)])