from naprima import ControlAdaptor


class KSPControlStrategy(ControlAdaptor):
    def __init__(self,timestep):
        self.timestep = timestep
        self.t = 0
        self.connection = krpc.connect()
        self.space_center = self.connection.space_center
        self.vessel = self.space_center.active_vessel
        arms = []
        for gripper in self.find_grippers(self.vessel):
            arms.append(self.find_motors(gripper))
        self.arm = arms[0]
        for motor in self.arm:
            motor.modules[0].set_field_float('Current Angle',0)
        self.signs = list(zeros((1,len(arms))))
        self.commandQueue = Queue()
        self.outQueue = Queue()
    
        def control_loop(in_q,out_q):
            while True:                
                command = in_q.get()
                if command is None:
                    pass
                if command[0].__code__.co_argcount == len(command)-1:
                    if(len(command)==2):
                        out_q.put(command[0](command[1]))
                    elif(len(command)==3):
                        command[0](command[1],command[2])
                else:
                    raise Exception("wrong number of parameters")
                
                for motor in self.arm:
                    if self.signs[self.arm.index(motor)] != 0:
                        self.set_motor_position(motor, self.get_motor_position(motor)+self.get_motor_target_velocity(motor))
                        
        self.T = t2 = Thread(target = control_loop, args =(self.commandQueue, self.outQueue, ))
        self.T.start()
                

    def find_grippers(self,vessel):
        return [p for p in vessel.parts.all if p.name == 'smallClaw']

    def find_motors(self,gripper):
        motors = []
        current = gripper
        while current is not None:
            if current.title == 'Rotation Servo M-06':
                motors.append(current)
            current = current.parent
        motors.reverse()
        return motors

    def set_motor_position(self,motor,position):
        mod = [m for m in motor.modules if m.name == 'ModuleRoboticRotationServo'][0]
        mod.set_field_float('Target Angle',rad2deg(position))

    def get_motor_position(self,motor):
        mod = [m for m in motor.modules if m.name == 'ModuleRoboticRotationServo'][0]
        print(mod.fields['Current Angle'])
        return deg2rad(float(mod.fields['Current Angle']))

    def set_motor_speed(self,motor,speed):
        mod = [m for m in motor.modules if m.name == 'ModuleRoboticRotationServo'][0]
        self.signs[self.arm.index(motor)] = sign(speed)
        mod.set_field_float('Traverse Rate',abs(rad2deg(speed)))
        
    def set_thetas(self,arm,thetas):
        if len(arm) == len(thetas):
            for motor,theta in zip(arm,thetas):
                self.set_motor_position(motor,theta)
        else:
            print("I'm afraid, I can't do that, Dave!")

    def set_theta_dot(self,arm,theta_dot):
        if len(arm) == len(theta_dot):
            for motor,theta_d in zip(arm,theta_dot):
                self.set_motor_speed(motor,rad2deg(theta_d))
        else:
            print("I'm afraid, I can't do that, Dave!")

    def get_motor_positions(self,arm):
        return [self.get_motor_position(motor) for motor in arm]

    #///////////////////////////////////////////////////////////////
    #///////////////////////////////////////////////////////////////
    
    def get_joint_positions(self):
        #self.get_motor_positions(self.arm) 
        self.commandQueue.put((self.get_motor_positions,self.arm))
        while self.outQueue.qsize() < 1:
            pass
        return self.outQueue.get()

    def get_joint_velocities(self):
        return 

    def set_joint_positions(self, positions):
        #self.set_thetas(self.arm,positions)
        self.commandQueue.put((self.set_thetas,self.arm,positions))

    def set_joint_velocities(self, velocities):
        pass