# The `naprima` Package
The `naprima` package is the result of a master's thesis project and was used to faciliate the evaluation of a mathematical model of arm movement assembly from movement primitives. Its structure suggests an interface against which modules for the kinematic modeling of robots, control interfaces for the operation of robots or robot simulations, perception modules and modules for the planning of movements can be developed and added to a the existing ones to provide a library for the modular design and execution of robotics experiments.

View the documentation [here](https://doertemithut.github.io/naprima/).<br>
If you have any suggestions, feel free to use the issue tracker.

# An Example Experiment
In the following, the use of the full application scope of the `naprima` package will be illustrated by setting up an example.

## Generating Movements from Movement Primitives
We start by defining the transport primitive
```python
def transport(t,task): 
    # extract needed information from task
    tau = task["tau"]
    p_target = cartesian(task["p_target"])
    p_0 = cartesian(task["p_0"])
    
    # compute linear component of primitive
    n = p_target-p_0
    norm_n = np.linalg.norm(n)
    n_hat = n/norm_n
    P_lin = n_hat*norm_n*(1-np.cos(2*t*np.pi/tau))/tau
    
    # assemble SE(3) matrix from P_lin
    return RpToTrans(np.eye(3),P_lin.T)
    #Note: RpToTrans is a function from the modern_robotics package
```
In the python file `main.py` we will be executing, we can now import the transport primitive as a function:
```python
from naprima.primitives import transport
```
In order to call the primitive, we need to define a task to supply it with the required information about the environment. A task is defined as a python dictionary in `main.py` as follows:

```python
# importing primitives
from naprima.primitives import transport

# defining initial positions of target and eef
p_target = [0,1,0.2]
p_0 = [0,0.5,0.2]

# defining a task as a dictionary
task = {
    "tau" : 1.5,
    "p_target": p_target,
    "p_0": p_0
}

# calling the primitive at instant 0.1
transport(0.1,task)

# producing a time series of the primitive's value
from numpy import linspace
values = [transport(t,task) for t in linspace(0,1.5,100)]
```
If we import a second primitive `vertical_lift` from `naprima.primitives`, which requires an amplitude parameter from the task, we can now import the movement strategy class to assemble both primitives into a movement strategy as follows:


```python
# importing primitives
from naprima.primitives import transport, vertical_lift
# importing Movement Strategy
from naprima.strategies import MovementStrategy

# defining initial positions of target and eef
p_target = [0,1,0.2]
p_0 = [0,0.5,0.2]

# adding lift_amplitude to the task
task = {
    "tau" : 1.5,
    "p_target": p_target,
    "p_0": p_0
    "lift_amplitude": 0.3
}

# instantiating a strategy to apply transport and lift to the task 
strategy = MovementStrategy(task,[transport,vertical_lift])

# obtaining movement at instant t as SE(3)-matrix from strategy 
strategy.get_movement(t,"SE(3)")
```

## Kinematics

To deploy the movements that result from a strategy to a kinematic model of a kinematic chain, we import the `KinematicChain` class from `naprima.robotics` and define a kinematic chain by providing the axes of rotation for its joints, the positions of said joints in the chosen home position and the SE(3)-matrix to transform the base frame {B} into the tool frame {T} when the robot is in its home position.

For illustration purposes I will use a 2R(meaning two revolute joints)-robot.

```python
# importing primitives
from naprima.primitives import transport, vertical_lift
from naprima.strategies import MovementStrategy
from naprima.robotics import KinematicChain

# import numpy for matrix operations and array class
import numpy as np

# defining rotation axes of joints
joint_axes = [[   0,  1,  0],[   0,  1, 0]]
joint_positions = [[0,0,0],[0,0,0.5]]
tool_frame = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,1.0],
                       [0,0,0,1]])

robot = KinematicChain(joint_axes,joint_positions,tool_frame)
                       
# defining initial positions of target and eef
p_target = [0,1,0.2]
# get initial eef position from the forward kinematic with thetas = (0,0)
current_theta = [0,0]
p_0 = robot.get_fk_position(current_theta)

# adding lift_amplitude to the task
task = {
    "tau" : 1.5,
    "p_target": p_target,
    "p_0": p_0
    "lift_amplitude": 0.3
}

# instantiating a strategy to apply transport and lift to the task 
strategy = MovementStrategy(task,[transport,vertical_lift])

# obtaining the spatial velocity at instant t from movement strategy 
twist = strategy.get_movement(t,"TwistVector")
# obtaining desired joint velocities by leftt multiplying the current Jacobian 
inv_Jacobian = robot.get_inverse_Jacobian(current_theta) @ twist
# updating current theta with a 1s-timestep
current_theta = current_theta + theta_dot
```