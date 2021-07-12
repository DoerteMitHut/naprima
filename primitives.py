import numpy as np
import scipy.linalg
from  scipy.spatial.transform import Rotation
import modern_robotics as mr
from naprima.utils import *

##////// TRANSPORT /////////
def transport (t,task):
    '''Linear primitive to transport the end effector to a target location in a straight line, following a sinusoid velocity profile.'''
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
    if t > task["tau"]:
        return mr.RpToTrans(np.eye(3),np.zeros([3,1]))
    return mr.RpToTrans(np.eye(3),P_lin.T)

def horizontal_transport (t,task):
    '''Linear primitive to transport the end effector in a horizontal straight line to the projection of a target location p_target onto the horizontal plane in which the initial end effector position p_0 is contained, following a sinusoid velocity profile.''' 
    # extract needed information from task
    tau = task["tau"]
    p_target = cartesian(task["p_target"])
    p_0 = cartesian(task["p_0"])
    
    
    # compute linear component of primitive
    n = p_target-p_0
    n[0,1] = 0
    norm_n = np.linalg.norm(n)
    n_hat = normalize(n)
    P_lin = n_hat*norm_n*(1-np.cos(2*t*np.pi/tau))/tau
    
    # assemble SE(3) matrix from P_lin
    if t > task["tau"]:
        return mr.RpToTrans(np.eye(3),np.zeros([3,1]))
    return mr.RpToTrans(np.eye(3),P_lin.T)

def vertical_transport (t,task):
    '''Linear primitive to transport the end effector in a horizontal straight line to the projection of a target location p_target onto the horizontal plane in which the initial end effector position p_0 is contained, following a sinusoid velocity profile.''' 
    # extract needed information from task
    tau = task["tau"]
    p_target = cartesian(task["p_target"])
    p_0 = cartesian(task["p_0"])
    
    # compute linear component of primitive
    n_hat = column([0,1,0])
    norm_n = p_target[0,1]-p_0[0,1]
    P_lin = n_hat*norm_n*(1-np.cos(2*t*np.pi/tau))/tau
    # assemble SE(3) matrix from P_lin
    if t > task["tau"]:
        return mr.RpToTrans(np.eye(3),np.zeros([3,1]))
    return mr.RpToTrans(np.eye(3),P_lin)

##////// LIFT /////////////
def lift(t,task): 
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        A = task["A"]
        alpha = task["alpha"]
    except KeyError as err:
                raise err
    
    n_hat = normalize(row(p_target-p_0))
    if np.linalg.norm(n_hat) < 10e-14:
        n_hat = row([1,0,0])
    #sideways = np.array([-direct[0,0]*direct[0,1],np.power(direct[0,0],2)+np.power(direct[0,2],2),-direct[0,1]*direct[0,2]])
    
    sideways = normalize(np.cross(listify(n_hat),[0,1,0]))
    
    R = Rotation.from_rotvec(-n_hat*alpha)
    p_dot_length = (A*np.pi/tau)*(np.sin(t*2*np.pi/tau))
    p_dot = R.apply(sideways)*p_dot_length
    return mr.RpToTrans(np.eye(3),p_dot.T)

def asymmetric_lift(t,task):
    '''Linear primitive to be combined with the direct transport primitive to displace the end effector along a vertical vector by a distance sinusoid with lift_periodiod tau.'''
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        alpha = task["alpha"]
        A = task["A"]
        t_mark = task["t_mark"]
    except KeyError as err:
        raise err
    y_hat = column([0,1,0])

    n = p_target-p_0
    n_norm = np.linalg.norm(n)
    n_hat = normalize(n)
    if np.linalg.norm(n_hat) <= 10e-14:
        n_hat = row([1,0,0])
    lift_vector =  scipy.linalg.expm(VecToso3(-alpha*n_hat)) @ normalize(column(np.cross(listify(n_hat),[0,1,0])))
    if t == 0:
        print("[primitive]:",A,alpha)
    # quintic polynomial approximating the instant when the encounter position is passed
    # t_mark = tau*(15.2338*np.power(d_encounter/n_norm,5) -38.0846*np.power(d_encounter/n_norm,4) + 35.2123*np.power(d_encounter/n_norm,3)  -14.7339*np.power(d_encounter/n_norm,2)  +  3.2534*np.power(d_encounter/n_norm,1)  +  0.0594)
    if t<=t_mark:
        #p_dot_length = A*(1-np.cos(2*t*np.pi/t_mark))/t_mark
        p_dot_length = (A*np.pi/(2*t_mark))*(np.sin(t*2*np.pi/(2*t_mark)))
    else:
        #p_dot_length = -A*(1-np.cos(2*t*np.pi/(tau-t_mark)))/(tau-t_mark)
        p_dot_length = -(A*np.pi/(2*(tau-t_mark)))*(np.sin((t-t_mark)*2*np.pi/(2*(tau-t_mark))))
    p_dot = column(lift_vector*p_dot_length)
    
    return mr.RpToTrans(np.eye(3),p_dot)

def asymmetric_lift_Rot(t,task):
    '''Linear primitive to be combined with the direct transport primitive to displace the end effector along a vertical vector by a distance sinusoid with lift_periodiod tau.'''
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        alpha = task["alpha"]
        A = task["A"]
        t_mark = task["t_mark"]
    except KeyError as err:
        raise err
    y_hat = column([0,1,0])

    n = p_target-p_0
    n_norm = np.linalg.norm(n)
    n_hat = normalize(n)
    if np.linalg.norm(n_hat) <= 10e-14:
        n_hat = row([1,0,0])
    sideways = normalize(np.cross(listify(n_hat),[0,1,0]))
    
    R = Rotation.from_rotvec(-n_hat*alpha)
    lift_vector = R.apply(sideways)
    if t == 0:
        print("[primitive]:",A,alpha)
    # quintic polynomial approximating the instant when the encounter position is passed
    # t_mark = tau*(15.2338*np.power(d_encounter/n_norm,5) -38.0846*np.power(d_encounter/n_norm,4) + 35.2123*np.power(d_encounter/n_norm,3)  -14.7339*np.power(d_encounter/n_norm,2)  +  3.2534*np.power(d_encounter/n_norm,1)  +  0.0594)
    if t<=t_mark:
        #p_dot_length = A*(1-np.cos(2*t*np.pi/t_mark))/t_mark
        p_dot_length = (A*np.pi/(2*t_mark))*(np.sin(t*2*np.pi/(2*t_mark)))
    else:
        #p_dot_length = -A*(1-np.cos(2*t*np.pi/(tau-t_mark)))/(tau-t_mark)
        p_dot_length = -(A*np.pi/(2*(tau-t_mark)))*(np.sin((t-t_mark)*2*np.pi/(2*(tau-t_mark))))
    p_dot = column(lift_vector*p_dot_length)
    
    return mr.RpToTrans(np.eye(3),p_dot)


    '''Linear primitive to be combined with the direct transport primitive to displace the end effector along a vertical vector by a distance sinusoid with lift_periodiod tau.'''
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        h_obst = task["h_obst"]
        p_obst = cartesian(task["p_obst"])
        r_obst = task["r_obst"]
        clearance = task["clearance"]


        evasion_parameter = 0.2
        steepness_parameter=200
    except KeyError as err:
                raise err

    y_hat = row([0,1,0])

    n = p_target-p_0
    n_norm = np.linalg.norm(n)
    n_hat = normalize(n)
    d_hat = np.cross(listify(n_hat),listify(y_hat))

    p_0_base = row([p_0[0][0],p_obst[0][1],p_0[0][2]])
    n_base = row([n[0][0],0,n[0][2]])
    AB = p_obst-p_0_base
    angle = np.arccos(np.dot(normalize(n_base),normalize(AB).T))[0][0]#why does this become an array?
    n_base_length = np.cos(angle)*np.linalg.norm(AB)
    n_slope = n[0][1]/np.linalg.norm(n_base)
    
    q_n = p_0 + normalize(n_base)*n_base_length + row([0,n_slope*n_base_length,0])    
    q_o = row([p_obst[0][0],q_n[0][1],p_obst[0,2]])
    #q_n = p_0   +   S1  *   n_hat

    #q_o = q_n+   d  *   d_hat

    S1 = np.linalg.norm(q_n-p_0)
    S2 = np.linalg.norm(q_o-p_obst)

    q_dist = np.linalg.norm(q_o-q_n)

    #print("dots: ", np.dot(n_hat,d_hat),np.dot(y_hat,d_hat))
    if q_dist < clearance+r_obst:
        
        d = q_dist *d_hat

        h_max_ele = h_obst-S2
        h_ele = h_max_ele*(-1/(1+np.exp(-(h_obst-(S2+evasion_parameter))*steepness_parameter))+1)
        p_pass_spine = q_o + y_hat * h_ele


        p_pass1 = p_pass_spine + d_hat * (clearance+r_obst)
        transport_slope = np.arcsin(n[0][1]/n_norm)
        transport_spine_angle = (np.pi/2)-transport_slope

        encounter_offset = h_ele * np.cos(transport_spine_angle)
        d_encounter = S1-encounter_offset
        zenith_base = p_0+(d_encounter)*n_hat
        p_pass = zenith_base+d_hat * (d+clearance*(1-(h_ele/h_max_ele))+r_obst)+(np.cross(d_hat,n_hat)*h_ele*np.sin(transport_spine_angle))
        if t == 0:
            print(p_0)
            print(h_ele/h_max_ele)
            print(p_pass)
        peak_lift_vector = p_pass-zenith_base
        #print(np.dot(normalize(peak_lift_vector),n_hat.T))
        #print(peak_lift_vector)
        lift_vector = normalize(peak_lift_vector)
        A = np.linalg.norm(peak_lift_vector)
        #elevation_angle = np.arccos(np.dot(lift_vector,d_hat))
        
        # quintic polynomial approximating the instant when the encounter position is passed
        t_mark = tau*(15.2338*np.power(d_encounter/n_norm,5) -38.0846*np.power(d_encounter/n_norm,4) + 35.2123*np.power(d_encounter/n_norm,3)  -14.7339*np.power(d_encounter/n_norm,2)  +  3.2534*np.power(d_encounter/n_norm,1)  +  0.0594)

        if t<=t_mark:
            #p_dot_length = A*(1-np.cos(2*t*np.pi/t_mark))/t_mark
            p_dot_length = (A*np.pi/(2*t_mark))*(np.sin(t*2*np.pi/(2*t_mark)))
        else:
            #p_dot_length = -A*(1-np.cos(2*t*np.pi/(tau-t_mark)))/(tau-t_mark)
            p_dot_length = -(A*np.pi/(2*(tau-t_mark)))*(np.sin((t-t_mark)*2*np.pi/(2*(tau-t_mark))))
        p_dot = column(lift_vector*p_dot_length)
    
    else:
        p_dot = column([0,0,0])

    return mr.RpToTrans(np.eye(3),p_dot)

##///// GRASP /////////////

def grasp_wrap_evo(t,task):
    '''Gripper deformation primitive for achieving a target aperture from an initial one.'''
    # extract needed information from task
    tau = task["tau"]
    Ap_target = task["Ap_target"]
    Ap_0 = task["Ap_0"]
    
    # compute linear component of primitive
    d = Ap_target-Ap_0
    a_dot = d*(1-np.cos(2*t*np.pi/tau))/tau

    # assemble SE(3) matrix from P_lin
    if t > task["tau"]:
        return np.array([0])
    return np.array([a_dot])

def grasp_wrap_open(t,task):
    '''Gripper deformation primitive for achieving a target aperture from an initial one.'''
    # extract needed information from task
    tau = task["tau"]
    Ap_max = task["Ap_max"]
    Ap_0 = task["Ap_0"]
    Ap_peak_t = task["Ap_peak_t"]
    
    # compute linear component of primitive
    d = Ap_max-Ap_0
    a_dot = d*(1-np.cos(2*t*np.pi/Ap_peak_t))/Ap_peak_t

    # assemble SE(3) matrix from P_lin
    if t > task["Ap_peak_t"]:
        return np.array([0])
    return np.array([a_dot])

def grasp_wrap_close(t,task):
    '''Gripper deformation primitive for achieving a target aperture from an initial one.'''
    # extract needed information from task
    tau = task["tau"]
    Ap_max = task["Ap_max"]
    Ap_min = task["Ap_min"]
    Ap_0 = task["Ap_0"]
    Ap_peak_t = task["Ap_peak_t"]
    
    # compute linear component of primitive
    d = Ap_min-Ap_max
    tau_close = tau-Ap_peak_t
    a_dot = d*(1-np.cos(2*t*np.pi/tau_close))/tau_close

    # assemble SE(3) matrix from P_lin
    if t <= task["Ap_peak_t"]:
        return np.array([0])
    return np.array([a_dot])

##////// ORIENTATION ///////

def hand_orientation_bell(t,task):
    omega =column(so3ToVec(scipy.linalg.logm(task["R_target"] @ task["R_0"].T)))*((1-np.cos(2*t*np.pi/task["tau"]))/task["tau"])
    p = column(task["KC"].get_fk_position(task["RC"].get_joint_positions()))
    twist = column(np.hstack([omega.T,row(np.cross(-omega.T,p.T))]))
    T = scipy.linalg.expm(VecTose3(twist))
    return T


