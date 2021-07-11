from naprima.primitives import *
from scipy.linalg import expm,logm
import numpy

class MovementStrategy:
    def __init__(self,primitives):
        self.primitives = primitives
    def get_movement_series(self,task,timestep,representation='SE(3)'):
        return [self.get_movement(t,task,representation) for t in np.arange(0,task["tau"],timestep)]

    def get_movement(self,t,task,representation='SE(3)'):
        if representation == "TwistVector":
            return se3ToVec(logm(self.get_transform(t,task))) 
        elif representation == "SE(3)":
            return self.get_transform(t,task)
        elif representation == "se(3)":
            return logm(self.get_transform(t,task))
        else:
            raise NotImplementedError("You requested a movement representation which is not implemented.")
    
    def get_transform(self,t,task):
        T = np.eye(4)
        for p in self.primitives:
            T = T @ expm(logm(p(t,task))*task["timestep"])
        return T

class DeformationStrategy:
    def __init__(self,primitives,M = 1):
        self.M = M # number of shape DoFs
        self.primitives = primitives
    def get_deformation_series(self,task,timestep):
        return [self.get_movement(t,task)for t in np.arange(0,task["tau"],timestep)]

    def get_deformation(self,t,task):
        gamma_dots = np.zeros(self.M)
        for p in self.primitives:
            gamma_dots += p(t,task)*task["timestep"]
        return gamma_dots



# class MovementStrategy:
#     def __init__(self,task,primitives):
#         self.primitives = primitives
#         self.task = task

#     def get_movement(self,t,representation='SE(3)'):
#         if representation == "TwistVector":
#             return se3ToVec(logm(self.get_transform(t))) 
#         elif representation == "SE(3)":
#             return self.get_transform(t)
#         elif representation == "se(3)":
#             return logm(self.get_transform(t))
#         else:
#             raise NotImplementedError("You requested a movement representation which is not implemented.")
    
#     def get_transform(self,t):
#         T = np.eye(4)
#         for p in self.primitives:
#             T = T @ expm(logm(p(t,self.task))*0.0016)
#         return T
