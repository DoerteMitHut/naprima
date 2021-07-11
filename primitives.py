import numpy as np
import scipy.linalg
from  scipy.spatial.transform import Rotation
import modern_robotics as mr
from naprima.utils import *

##///////////// FINAL /////////////////
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

def lift_parameter(t,task):
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        h_obst = task["h_obst"]
        p_obst = cartesian(task["p_obst"])
        r_obst = task["r_obst"]
        clearance = task["clearance"]
    except KeyError as err:
        raise err

    y_hat = row([0,1,0])

    # transport vector
    n = p_target-p_0
    n_norm = np.linalg.norm(n)
    n_hat = normalize(n)
    if np.linalg.norm(n_hat) < 10e-14:
        n_hat = row([1,0,0])

    #direction of the shortest distance between transport line and obstacle spine
    d_hat = np.cross(listify(n_hat),listify(y_hat))

    #projection image of p0 and vector to obstacle base
    p_0_base = row([p_0[0][0],p_obst[0][1],p_0[0][2]])
    AB = p_obst-p_0_base

    #projection of n_trans
    n_base = row([n[0][0],0,n[0][2]])
    
    
    angle = np.arccos(np.dot(normalize(n_base),normalize(AB).T))[0][0]#why does this become an array?
    n_base_length = np.cos(angle)*np.linalg.norm(AB)
    n_slope = n[0][1]/np.linalg.norm(n_base)
    
    q_n = p_0 + normalize(n_base)*n_base_length + row([0,n_slope*n_base_length,0])    
    q_o = row([p_obst[0][0],q_n[0][1],p_obst[0,2]])

    S1 = np.linalg.norm(q_n-p_0)
    S2 = np.linalg.norm(q_o-p_obst)

    q_dist = np.linalg.norm(q_o-q_n)

    if q_dist < clearance+r_obst:
        
        d = q_dist *d_hat

        h_max_ele = h_obst-S2
        h_ele = h_max_ele*(-1/(1+np.exp(-(h_obst-(S2+evasion_parameter))*steepness_parameter))+1)
        p_pass_spine = q_o + y_hat * h_ele


        p_pass = p_pass_spine + d_hat * (clearance+r_obst)
        transport_slope = np.arcsin(n[0][1]/n_norm)
        transport_spine_angle = (np.pi/2)-transport_slope

        encounter_offset = h_ele * np.cos(transport_spine_angle)
        d_encounter = S1-encounter_offset
        zenith_base = q_n
        p_pass1 = zenith_base+d_hat * (d+clearance*(1-(h_ele/h_max_ele))+r_obst)+(np.cross(d_hat,n_hat)*h_ele*np.sin(transport_spine_angle))

        peak_lift_vector = p_pass-zenith_base
        #print(np.dot(normalize(peak_lift_vector),n_hat.T))
        #print(peak_lift_vector)
        lift_vector = normalize(peak_lift_vector)
        A = np.linalg.norm(peak_lift_vector)
        langle = np.arccos(np.dot(lift_vector,d_hat))

    else:
        A = 0
        langle = 0
    
    return (A,langle)

def asymmetric_lift_parameter(t,task):
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


##///////// LEGACY //////////////

def vertical_obstacle_avoidance(t,task):
    '''Linear primitive to be combined with the direct transport primitive to vertically avoid an obstacle along the path, the postion p_obst and height h_obst are known. The obstacle is avoided with a distance defined by the clearance parameter c_obst'''
    try:
        tau = task["tau"]
        lift_peak = task["lift_peak"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        p_obst=task["p_obst"]
        h_obst = task["h_obst"]
        c_obst = task["c_obst"]
    except KeyError as err:
                raise err

    # transport vector
    direct = p_target-p_0
    norm_direct= np.linalg.norm(direct)
    normalized_direct = direct/norm_direct
    
    # horizontal projection of transport vector
    direct_proj = direct
    direct_proj[0,1] = 0
    norm_direct_proj = np.linalg.norm(direct_proj)

    # vector between p_0 and p_obst
    p_0_obst = row(p_obst)-row(p_0)
    norm_p_0_obst = np.linalg.norm(p_0_obst)
    normalized_p_0_obst = p_0_obst/norm_p_0_obst

    # portion of transport distance at which the obstacle is encountered
    lift_peak= np.dot(listify(normalized_direct),listify(normalized_p_0_obst))*norm_p_0_obst/norm_direct
    P = tau * np.arccos(-lift_peak*2+1)/4+0.05*(lift_peak-0.35)+tau*.1
    lift_period = 0.4
    #P = 0.6

    # direction of lift
    lift_vector = np.array([0,1,0])
    normalized_lift_vector = lift_vector / np.linalg.norm(lift_vector)

    # amplitude of lift
    adj = lift_peak*norm_direct_proj
    hyp = lift_peak*norm_direct
    d_h = np.sin(np.arccos(adj/hyp))*hyp
    coll = p_obst[1]+h_obst-(p_0[0,1]-(d_h+c_obst))#
    amp_domain_limit = (1./(1+np.exp(-(coll)*100)))
    amp = amp_domain_limit*coll


    # magnitude of linear velocity along lift vector
    lift_magnitude = (amp*np.pi/lift_period)*(np.sin((t-(P-lift_period/2))*2*np.pi/lift_period))

        # limit lift domain with overlapnp.ping sigmoids
    right_limit = P+lift_period*.5
    left_limit = P-lift_period*.44
    limit_steepness = 100
    left_limit_sigmoid = (1./(1+np.exp(-(t-left_limit)*limit_steepness)))
    right_limit_sigmoid = (1-1./(1+np.exp(-(t-right_limit)*limit_steepness))) 
    domain_limit = left_limit_sigmoid * right_limit_sigmoid

    # velocity magnitude
    p_dot_length = domain_limit*lift_magnitude
    #p_dot_length = lift_magnitude
    
    p_dot = normalized_lift_vector*p_dot_length
    return mr.RpToTrans(np.eye(3),p_dot.T)

def vertical_obstacle_avoidance_h_trans(t,task):
    '''Linear primitive to be combined with the '''
    try:
        tau = task["tau"]
        lift_peak = task["lift_peak"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        h_obst = task["h_obst"]
    except KeyError as err:
                raise err
    P = tau*lift_peak
    direct = p_target-p_0
    direct[0,1] = 0
    lift_v = np.array([0,1,0])
    p_dot_length = (1-1./(1+np.exp(-(t-P)*100)))*(h_obst*np.pi/P)*(np.sin(t*2*np.pi/P))
    p_dot = (lift_v / np.linalg.norm(lift_v))*p_dot_length
    return mr.RpToTrans(np.eye(3),p_dot.T)

def lift_h_trans(t,task):
    '''Linear primitive to be combined with the horizontal transport primitive to displace the end effector along a vertical vector by a distance sinusoid with lift_periodiod tau'''
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        h_obst = task["h_obst"]
    except KeyError as err:
                raise err

    direct = p_target-p_0
    direct[0,1] = 0
    r = np.array([np.random.random() for i in range(3)])
    lift_v = np.array([0,1,0])
    p_dot_length = (h_obst*np.pi/tau)*(np.sin(t*2*np.pi/tau))
    p_dot = (lift_v / np.linalg.norm(lift_v))*p_dot_length
    return mr.RpToTrans(np.eye(3),p_dot.T)

def vertical_lift(t,task):
    '''Linear primitive to be combined with the direct transport primitive to displace the end effector along a vertical vector by a distance sinusoid with lift_periodiod tau.'''
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        h_obst = task["h_obst"]
    except KeyError as err:
                raise err

    direct = p_target-p_0
    direct[0,1] = 0
    r = np.array([np.random.random() for i in range(3)])
    lift_v = np.array([0,1,0])
    p_dot_length = (h_obst*np.pi/tau)*(np.sin(t*2*np.pi/tau))
    p_dot = (normalize(lift_v))*p_dot_length
    return mr.RpToTrans(np.eye(3),p_dot.T)

def grasp_naive(t,task):
    Amax = np.pi/2
    A0 = np.pi/4
    Atau = np.pi/8
    velocities = [-np.pi,-np.pi,-np.pi]
    if t>0.9*task["tau"]:
        velocities = [0,0,0]
    return velocities
import timeit
def grasp_wrap(t,task):
    tau = task["tau"]
    tpass = 0.8

    D = 0.057158
    a = 0.0685
    b = 0.0865
    beta = np.pi/16
    gamma = np.arctan(np.sin(beta)/np.cos(beta))
    d = a*np.sin(beta)/np.sin(gamma)
    tic = timeit.default_timer()
    amp_dot = (t*np.pi*np.sin((np.power(t,2)*np.pi*(- np.power(tau,3) + t*np.power(tau,2) + 2*np.power(tpass,3) - 2*t*np.power(tpass,2)))/(np.power(tau,2)*np.power(tpass,2)*(tau - tpass)))*(- 2*np.power(tau,3) + 3*t*np.power(tau,2) + 4*np.power(tpass,3) - 6*t*np.power(tpass,2)))/(np.power(tau,3)*np.power(tpass,2)*(tau - tpass))
    toc = timeit.default_timer()
    #print("time: ",toc-tic)
    alpha_dot = np.arccos(-amp_dot/2)

    #Amax = np.pi/2
    #A0 = np.pi/4
    #Atau = np.pi/8
    velocities = [alpha_dot,alpha_dot,alpha_dot]
    if t>=0:#task["tau"]:
        velocities = [-np.pi/4,-np.pi/4,-np.pi/4]
    return velocities

def vertical_transport_sin (t,task):
    '''Linear primitive to transport the end effector in a horizontal straight line to the projection of a target location p_target onto the horizontal plane in which the initial end effector position p_0 is contained, following a sinusoid velocity profile.''' 
    # extract needed information from task
    tau = task["tau"]
    p_target = cartesian(task["p_target"])
    p_0 = cartesian(task["p_0"])
    
    

    # compute linear component of primitive
    n = column([0,1,0])
    norm_n = p_target[0,1]-p_0[0,1]
    p_dot_length = (norm_n*np.pi/(2*tau))*(np.sin(t*2*np.pi/(2*tau)))
    P_lin = n*p_dot_length
    # assemble SE(3) matrix from P_lin
    if t > task["tau"]:
        return mr.RpToTrans(np.eye(3),np.zeros([3,1]))
    return mr.RpToTrans(np.eye(3),P_lin)

def hand_orientation(t,task):
    omega =column(so3ToVec(scipy.linalg.logm(task["R_target"] @ task["R_0"].T)))/task["tau"]#column([1,0,0])
    p = column(task["KC"].get_fk_position(task["RC"].get_joint_positions()))
    twist = column(np.hstack([omega.T,row(np.cross(-omega.T,p.T))]))
    T = scipy.linalg.expm(VecTose3(twist))
    return T

def hand_orientation_bell(t,task):
    omega =column(so3ToVec(scipy.linalg.logm(task["R_target"] @ task["R_0"].T)))*((1-np.cos(2*t*np.pi/task["tau"]))/task["tau"])
    p = column(task["KC"].get_fk_position(task["RC"].get_joint_positions()))
    twist = column(np.hstack([omega.T,row(np.cross(-omega.T,p.T))]))
    T = scipy.linalg.expm(VecTose3(twist))
    return T

def hand_orientation_bell_inplace(t,task):
    omega =column(so3ToVec(scipy.linalg.logm(task["R_target"] @ task["R_0"].T)))*((1-np.cos(2*t*np.pi/task["tau"]))/task["tau"])
    p = task["p_0"]
    twist = column(np.hstack([omega.T,row(np.cross(-omega.T,p.T))]))
    T = scipy.linalg.expm(VecTose3(twist))
    return T

def random_lift(t,task):
    '''Linear primitive to displace the end effector along the direct transport path along a random vector from the plane to which the transport vector is normal. Follows a sinusoid magnitude with lift_periodiod tau.'''
    try:
        tau = task["tau"]
        p_target = cartesian(task["p_target"])
        p_0 = cartesian(task["p_0"])
        h_obst = task["h_obst"]
    except KeyError as err:
                raise err

    direct = p_target-p_0
    direct[0,1] = 0
    r = np.array([np.random.random() for i in range(3)])
    lift_v = np.cross(direct,r)
    if lift_v[0,2] < 0:
        lift_v = np.cross(r,direct)
    p_dot_length = h_obst*(np.cos(t*2*np.pi/tau))
    p_dot = (lift_v / np.linalg.norm(lift_v))*p_dot_length
    return mr.RpToTrans(np.eye(3),p_dot.T)

def vertical_transport (t,task):
    '''Linear primitive to transport the end effector in a horizontal straight line to the projection of a target location p_target onto the horizontal plane in which the initial end effector position p_0 is contained, following a sinusoid velocity profile.''' 
    # extract needed information from task
    tau = task["tau"]
    p_target = cartesian(task["p_target"])
    p_0 = cartesian(task["p_0"])
    
    # compute linear component of primitive
    n = column([0,1,0])
    norm_n = p_target[0,1]-p_0[0,1]
    P_lin = n*norm_n/tau
    # assemble SE(3) matrix from P_lin
    if t > task["tau"]:
        return mr.RpToTrans(np.eye(3),np.zeros([3,1]))
    return mr.RpToTrans(np.eye(3),P_lin)


