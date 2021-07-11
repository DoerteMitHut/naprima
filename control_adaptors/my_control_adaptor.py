from naprima import ControlAdaptor

class MyControlAdaptor(ControlAdaptor):
    def __init__(self,robot_specs):
        self.robot = robot_specs
        super().__init__()
    def get_joint_positions(self):
        # Your work goes here
        raise NotImplementedError("There's work to be done. If you want to interface with a new kind of robot, please implement the methods of your control adaptor.")
    
    def get_joint_velocities(self):
        # Your work goes here
        raise NotImplementedError("There's work to be done. If you want to interface with a new kind of robot, please implement the methods of your control adaptor.")

    def set_joint_positions(self, positions):
        # Your work goes here
        raise NotImplementedError("There's work to be done. If you want to interface with a new kind of robot, please implement the methods of your control adaptor.")

    def set_joint_velocities(self, velocities):
        # Your work goes here
        raise NotImplementedError("There's work to be done. If you want to interface with a new kind of robot, please implement the methods of your control adaptor.")