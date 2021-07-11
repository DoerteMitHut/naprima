from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class RobotController():
    def __init__(self, strategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def get_joint_positions(self):
        return self._strategy.get_joint_positions()
    
    def get_joint_velocities(self):
        return self._strategy.get_joint_velocities()
    
    def set_joint_positions(self, positions):
        self._strategy.set_joint_positions(positions)
    
    def set_joint_velocities(self, velocities):
        self._strategy.set_joint_velocities(velocities)


class ControlAdaptor(ABC):
    @abstractmethod
    def get_joint_positions(self):
        pass
    
    @abstractmethod
    def get_joint_velocities(self):
        pass
    
    @abstractmethod
    def set_joint_positions(self, positions):
        pass
    
    @abstractmethod
    def set_joint_velocities(self, velocities):
        pass
