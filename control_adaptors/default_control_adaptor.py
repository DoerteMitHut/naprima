import numpy as np
from naprima import ControlAdaptor

class DefaultControlAdaptor(ControlAdaptor):
    '''The most basic control adaptor. Keeps track of joint angles and/or -velocities and can integrate the current constant joint velocities over a specified timestep'''
    def __init__(self,joint_positions,timestep=None):
        self.initial_joint_positions = np.array(joint_positions).flatten()
        self.joint_positions = self.initial_joint_positions
        self.joint_velocities = np.zeros(len(joint_positions))
        self.timestep = timestep
    def get_joint_positions(self):
        return self.joint_positions

    def get_joint_velocities(self):
        return self.joint_velocities

    def set_joint_positions(self, positions):
        self.joint_positions = positions

    def set_joint_velocities(self, velocities):
        self.joint_velocities = velocities

    #======

    def get_timestep(self):
        return self.timestep

    def tick(self):
        #TODO appends instead of element-wise adding.
        raise NotImplementedError("The simulation tick of the DefaultControlAdaptor is broken and cannot be used.")
        self.joint_positions += self.joint_velocities * self.timestep

    def reset(self):
        self.__init__(self.initial_joint_positions,timestep=self.timestep)

