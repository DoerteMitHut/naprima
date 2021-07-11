from naprima import ControlAdaptor
from numpy import abs, array, cross, eye, vstack, hstack, zeros, sin, cos, pi, linalg, deg2rad, rad2deg, sign

class WebotsCAREN(ControlAdaptor):
    def __init__(self,robot):
        self.robot = robot
        self.armMotors = [robot.getDevice("arm_motor_%d"%(i)) for i in range(7)]
        self.armMotorPositionSensors = [m.getPositionSensor() for m in self.armMotors]
        for m in self.armMotorPositionSensors:
            m.enable(int(self.robot.getBasicTimeStep()))
        try:
            self.finger_motors = [(self.robot.getDevice("finger_motor_%d" % (i)), self.robot.getDevice("finger_tip_motor_%d" % (i))) for i in range(1, 4)]
            self.fingerMotorPositionSensors = [m[0].getPositionSensor() for m in self.fingerMotors]
            for m in self.fingerMotorPositionSensors:
                m.enable(int(self.robot.getBasicTimeStep()))
        except Exception:
            print("No finger motors registered")
        super().__init__()

    def get_joint_positions(self):
        positions = []
        for sensor in self.armMotorPositionSensors:
            positions.append(sensor.getValue())
        return positions
    
    def get_joint_velocities(self):
        velocities = []
        for motor in self.armMotors:
            velocities.append(motor.getPosition())
        return velocities

    def set_joint_positions(self, positions):
        for i in range(7):
            self.armMotors[i].setPosition(positions[i])
            self.armMotors[i].setVelocity(float(1))

    def set_joint_velocities(self, velocities):
        if max(abs(velocities)) > 1.95:
            velocities = velocities*(1.95/max(abs(velocities)))
        for i in range(7):
            self.armMotors[i].setPosition(float('inf'))
            self.armMotors[i].setVelocity(float(velocities[i]))

#//////////////////////////////////////////////////////////////////////
#//////////// NON-INTERFACE FUNCTIONS /////////////////////////////////
#//////////////////////////////////////////////////////////////////////

    def set_finger_joint_positions(self,positions):
        for i,fm in enumerate(self.finger_motors):
            fm[0].setPosition(positions[i])
            fm[0].setVelocity(1)
    
    def set_finger_joint_velocities(self,velocities):
        for i,fm in enumerate(self.finger_motors):
            fm[0].setPosition('inf')
            fm[0].setVelocity(velocities[i])
    
    def get_finger_joint_positions(self):
        positions = []
        for sensor in self.fingerMotorPositionSensors:
            positions.append(sensor.getValue())
        return positions
    
    def get_finger_joint_velocities(self):
        velocities = []
        for motor in self.armMotors:
            velocities.append(motor.getPosition())
        return velocities
