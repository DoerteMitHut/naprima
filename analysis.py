import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rc
from naprima.utils import *
import os
from pickle import loads, dumps
from multiprocessing import Process
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm,logm
class webots_position_reader:
    def __init__(self,node):
        self.node = node
        self.heading = "pos"
    
    def get_heading(self):
        return self.heading

    def get_value(self):
        return self.node.getPosition()

class webots_velocity_reader:
    def __init__(self,node,lin = True, scalar = False):
        self.node = node
        self.heading = "vel"
        self.linear = lin
        self.scalar = scalar
    def get_heading(self):
        return self.heading
    
    def get_value(self):
        if self.linear:
            if self.scalar:
                return np.linalg.norm(self.node.getVelocity()[0:3])
            else:
                return self.node.getVelocity()[0:3]
        else:
            if self.scalar:
                return np.linalg.norm(self.node.getVelocity()[3:])
            else:
                return self.node.getVelocity()[3:]

class DataDump:
    def __init__(self,readings,robot=None,rc=None,delimiter="|",break_lines=True,output_dir="."):
        self.readings = readings
        self.robot = robot
        self.rc = rc
        self.delimiter = delimiter
        self.break_lines= break_lines
        self.output_dir = output_dir
        self.data = {}

        with open("\\".join([output_dir,f"experiment_{counter}_task"]),'w') as output_task:
            self.output_task.write(json.dumps(self.task))
        
        with [open(f"experiment_{count}_{k}","w") for k in self.readings.keys()] as files:
            for i in range(len(readings)):
                files[i].write(self.readings[self.readings.keys()[i]]())
    
        output_target_position.write(str(task["target_pos"]))
        output_start_position.write(str(task["eef_origin"]))

        output_position = open(f"experiment_{counter}_position",'w')
        output_lin_vel = open(f"experiment_{counter}_lin_vel",'w')
        output_ang_vel = open(f"experiment_{counter}_ang_vel",'w')

    def get_headings(self):
        return self.delimiter.join(["t"]+[r.get_heading() for r in self.readings])

    def get_line(self,t):
        toRet = str(t)+self.delimiter+self.delimiter.join([str(r.get_value()) for r in self.readings])
        if self.break_lines:
            toRet = toRet + "\n"
        return toRet 

class DataDumpster:
    def __init__(self,eef_node,RC,task,strategy):        
        try:
            with open("counterfile",'r') as f:
                pass
        except:
            with open("counterfile",'w') as counterfile:
                counterfile.write(str(0))    
        with open("counterfile",'r') as counterfile:
            self.counter = int(counterfile.readline())+1
        with open("counterfile",'w') as counterfile:
            counterfile.write(str(self.counter))
        
        self.eef_node = eef_node
        self.RC = RC
        self.task = task
        self.strategy = strategy
        self.data = {
        "eef_origin": list(self.task["eef_origin"].flatten()),
        "target_pos": list(self.task["target_pos"].flatten()),
        "obstacle_pos": None,
        "obstacle_height": None,
        "tau" : self.task["tau"],
        "t" : [],
        "eef_pos" : [],
        "eef_fk" : [],
        "eef_lin_vel" : [],
        "eef_ang_vel" : [],
        "eef_lin_vel_desired": [],
        "eef_ang_vel_desired": [],
        "theta":[],
        "thetadot": [],
        "transport" : [],
        "lift" : []
    }
    
    def append(self,t,theta_dot):
        self.data["t"].append(t)
        self.data["eef_pos"].append(self.eef_node.getPosition())
        self.data["eef_fk"].append(list(FK(self.RC.get_joint_positions())[0:3].flatten()))
        self.data["eef_lin_vel"].append(self.eef_node.getVelocity()[0:3])
        self.data["eef_ang_vel"].append(self.eef_node.getVelocity()[3:])
        self.data["eef_lin_vel_desired"].append(list(self.strategy.get_transform(t)[0:3,3]))
        rot = self.strategy.get_transform(t)[0:3,0:3]
        self.data["eef_ang_vel_desired"].append([list(x) for x in list(rot)])
        self.data["theta"].append(list(self.RC.get_joint_positions()))
        self.data["thetadot"].append(list(theta_dot))
        for p in self.strategy.primitives:
            try:
                self.data[p.__name__].append(list(p(t,self.task)[0:3,3]))
            except Exception:
                self.data[p.__name__] = []
                self.data[p.__name__].append(list(p(t,self.task)[0:3,3]))

    def dump(self):
        with open(f"output\\output_{self.counter}",'w') as f:
            f.write(json.dumps(self.data))

class Plotter:
    def __init__(self,KC = None, RC = None, task=None, empty = False):
        if not empty:
            self.KC = KC
            self.RC = RC
            self.task = task
            self.it = 0
            self.t_series = [0]
            self.timestep = RC.get_timestep()
            self.velocity_series = [0]
            self.position_series = [column(cartesian(task['p_0']))]
            self.position_integrated_series = [column(cartesian(task['p_0']))]
            self.curvature_series = [0]
            self.twist_series = []
            self.movement_series = []
            self.rotation_series = []
            self.joint_position_series = [RC.get_joint_positions()] 
            self.joint_velocity_series = [RC.get_joint_velocities()]
            self.true_joint_velocity_series = [RC.get_joint_velocities()]
            self.SE3_series = [KC.get_fk_orientation(RC.get_joint_positions())]
            self.initial_state = self.__dict__
        rc('text', usetex=True)

    def register_KC(self,KC):
        self.KC = KC
    
    def register_RC(self,RC):
        self.RC = RC
    
    def register_task(self,task):
        self.task = task

    def log(self,**kwargs):
        self.it += 1
        self.t_series.append(kwargs['t'])
        self.twist_series.append(kwargs['twist'])
        self.position_integrated_series.append(column(cartesian(expm(VecTose3(kwargs['twist'])) @ column(homogenous(self.position_integrated_series[-1],position=True)))))
        self.joint_velocity_series.append(listify(kwargs['theta_dot']))
        self.true_joint_velocity_series.append(self.RC.get_joint_velocities())
        self.joint_position_series.append(listify(self.RC.get_joint_positions()))
        self.position_series.append(column(self.KC.get_fk_position(self.RC.get_joint_positions())))
        self.rotation_series.append(self.KC.get_fk_orientation(self.RC.get_joint_positions()))
        self.movement_series.append(kwargs["strategy"].get_movement(kwargs["t"],self.task,'TwistVector'))
        if self.it >= 2:
            self.velocity_series.append(np.linalg.norm(self.position_series[-1][:]-self.position_series[-2][:])/self.timestep)
            self.SE3_series.append(kwargs["KC"].get_fk_SE3(kwargs["RC"].get_joint_positions()))
    
    def plot_trajectory(self,ax = None, show=False,**kwargs):
        if not ax:
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')
        #change index to i,0
        ax.plot(*[[p[i,0] for p in self.position_series] for i in [2,0,1]],**kwargs)
        if show:
            plt.show()
        else:
            return ax

    def plot_trajectory_components(self,ax=None,show=False):
        plt.clf()
        ax = plt.gca()
        plt.ylabel(r"end effector position $\left[ m \right]$",fontsize=20)
        plt.xlabel(r"$t\left[s\right]$",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.plot(self.t_series,[p[0] for p in self.position_series],color='red')
        plt.plot(self.t_series,[p[1] for p in self.position_series],color='green')
        plt.plot(self.t_series,[p[2] for p in self.position_series],color='blue')
        plt.legend([r"".join([r"$",f"{axis}",r"$-position"]) for axis in ['x','y','z']],loc='upper left',bbox_to_anchor=(1.02,1),fontsize=15)
        if show:
            plt.show()
        else:
            return ax

    def plot_curvature(self,ax=None,show=False):
        if not ax:
            plt.clf()
            ax = plt.gca()
        plt.ylabel(r"r. o. curvature $\left[ \frac{1}{m} \right]$")
        plt.xlabel(r"$t\left[s\right]$")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        kappa = [np.linalg.norm((normalize(self.position_series[i+1])-2*normalize(self.position_series[i])+normalize(self.position_series[i-1]))/np.power(np.linalg.norm(self.position_series[i]-self.position_series[i-1])+np.linalg.norm(self.position_series[i+1]-self.position_series[i]),2)) for i in range(1,len(self.position_series)-1)]
        ax.plot(np.linspace(0,self.task["tau"],len(kappa)),kappa)
        plt.ylim(0,20)
        if show:
            plt.show()
        else:
            return ax

    def plot_velocity_by_curvature(self,ax=None,show=False):
        if not ax:
            plt.clf()
            ax = plt.gca()
        plt.ylabel(r"r. o. curvature $\left[ \frac{1}{m} \right]$")
        plt.xlabel(r"$t\left[s\right]$")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        kappa = [np.linalg.norm((normalize(self.position_series[i+1])-2*normalize(self.position_series[i])+normalize(self.position_series[i-1]))/np.power(np.linalg.norm(self.position_series[i]-self.position_series[i-1])+np.linalg.norm(self.position_series[i+1]-self.position_series[i]),2)) for i in range(1,len(self.position_series)-1)]
        ax.plot(np.linspace(0,self.task["tau"],len(kappa)),kappa)
        vel= self.velocity_series[int((len(self.velocity_series)-len(kappa))/2):int((len(self.velocity_series)-len(kappa))/2+len(kappa))]
        ax.plot(np.linspace(0,self.task["tau"],len(vel)),[v/np.power(k,1/3) for v,k in zip(vel,kappa)])
        plt.ylim(0,20)
        plt.legend(["curvature $\kappa\ [m]$","end effector speed $[\frac{m}{s}]$"])
        if show:
            plt.show()
        else:
            return ax
    
    def plot_torsion(self,ax=None,show=False):
        if not ax:
            plt.clf()
            ax = plt.gca()
        normals = [normalize((normalize(self.position_series[i+1])-2*normalize(self.position_series[i])+normalize(self.position_series[i-1]))/np.power(np.linalg.norm(self.position_series[i]-self.position_series[i-1])+np.linalg.norm(self.position_series[i+1]-self.position_series[i]),2)) for i in range(1,len(self.position_series)-1)]
        #print("Normals:",normals[0])
        tangents = [normalize(self.position_series[i+1]-self.position_series[i]) for i in range(1,len(self.position_series)-1)]
        binormals = [normalize(np.cross(listify(tangents[i]),listify(normals[i]))) for i in range(len(normals))]
        binormal_dots = [np.linalg.norm((binormals[i+1]-binormals[i])/(np.linalg.norm(self.position_series[i+2]-self.position_series[i+1]))) for i in range(len(binormals)-1)]
        
        ax.plot(np.linspace(self.timestep,self.task["tau"]-self.timestep,len(binormal_dots)),binormal_dots)
        if show:
            plt.show()
        else:
            return ax
    
    def plot_integrated_trajectory(self,ax = None, show=False,**kwargs):
        if not ax:
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')
        print(self.position_integrated_series[0][2])
        x = [p[0] for p in self.position_integrated_series]
        print(x)
        y = [p[1] for p in self.position_integrated_series]
        z = [p[2] for p in self.position_integrated_series]
        ax.plot(z,x,y)
        if show:
            plt.show()
        else:
            return ax

    def plot_base_frame(self,ax=None,show=False):
        if not ax:
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')

        x = (column([1,0,0,0])).flatten()
        y = (column([0,1,0,0])).flatten()
        z = (column([0,0,1,0])).flatten()

        for c,q in zip(['red','green','blue'],[x,y,z]):
            ax.quiver(0,0,0,q[2],q[0],q[1],color = c, length=.13, lw=1.5)
        if show:
            plt.show()
        else:
            return ax

    def plot_tool_frame(self,ax =None,i = None,show=False):
        if not ax:
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')
        #M = self.SE3_series[i]
        M = self.KC.get_fk_SE3(self.joint_position_series[i])
        print(M)
        p = homogenous(self.position_series[i])
        print(p)
        # z = listify(M @ column(homogenous([1,0,0])))
        # x = listify(M @ column(homogenous([0,1,0])))
        # y = listify(M @ column(homogenous([0,0,1])))
        print(column(homogenous([.1,0,0])))
        x = listify(M@column(homogenous([.05,0,0],position=True)))
        y = listify(M@column(homogenous([0,.05,0],position=True)))
        z = listify(M@column(homogenous([0,0,.05],position=True)))
        ax.plot([p[2],x[2]],[p[0],x[0]],[p[1],x[1]],color='r')
        ax.plot([p[2],y[2]],[p[0],y[0]],[p[1],y[1]],color='g')
        ax.plot([p[2],z[2]],[p[0],z[0]],[p[1],z[1]],color='b')
        #for c,x in zip(['green','blue','red'],[x,y,z]):
        #    ax.plot([p[2],x[2]],[p[0],x[0]],[p[1],x[1]],color=c)
        if show:
            plt.show()
        else:
            return ax

    def plot_arm(self,ax = None,i = 0,color='k',show=False, baseframe=False, toolframe=False):
        if not ax:
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')
        theta = listify(self.joint_position_series[i])
        arm = np.array([joint_KC.get_fk_position(theta).flatten() for i,joint_KC in enumerate(self.KC.joint_KCs)]).T
        ax.plot(arm[2,:],arm[0,:],arm[1,:],linestyle='-', marker='o', color=color,markersize=3,zorder=3)
        ax.plot(arm[2,-1],arm[0,-1],arm[1,-1],linestyle='-', marker='o', color='g',zorder=3)
        if baseframe:
            self.plot_base_frame(ax=ax)
        if toolframe:
            self.plot_tool_frame(ax=ax,i=i)
        if show:
            plt.show()
        else:
            return ax
        
    def plot_arm_snapshots(self,ax = None, show = False,**kwargs):
        if not ax:
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')
        try:
            color = kwargs['color']
        except Exception:
            color = 'k'
        
        if 'n' in kwargs.keys():
            for i in [int(np.floor(k)) for k in np.linspace(0,self.it-1,kwargs['n'])]:
                self.plot_arm(ax=ax,i=i,color=color)
        if show:
            plt.show()
        else:
            return ax

    def plot_arm_full(self,ax = None,i = 0,color='k',show=False, baseframe=False, toolframe=False):
        try:
            ax = self.plot_box(self.task["p_obst"][0,0],self.task["p_obst"][2,0],self.task["xd_obst"],self.task["zd_obst"],self.task["h_obst"],h_table=self.task["p_obst"][1,0])
        except:
            ax = plt.figure().add_subplot(1,1,1,projection='3d')
            print("no obstacle plotted")
        ax = self.plot_arm(ax=ax,i=0)
        ax = self.plot_trajectory(ax=ax)

        xmin = min([*(ax.get_xlim()),*(ax.get_ylim()),*(ax.get_zlim())])
        xmax = max([*(ax.get_xlim()),*(ax.get_ylim()),*(ax.get_zlim())])
        ax.set_xlim3d(left=xmin, right=xmax)
        ax.set_ylim3d(bottom=xmin, top=xmax)
        ax.set_zlim3d(bottom=xmin, top=xmax)
        if show:
            plt.show()
        else:
            return ax   

    def plot_thetas(self,ax = None,show=False):
        plt.clf()
        ax = plt.gca()
        
        ax.plot(self.t_series, [t[0] for t in self.joint_position_series])
        ax.plot(self.t_series, [t[2] for t in self.joint_position_series])
        ax.plot(self.t_series, [t[3] for t in self.joint_position_series])
        ax.plot(self.t_series, [t[1] for t in self.joint_position_series])
        ax.plot(self.t_series, [t[4] for t in self.joint_position_series])
        ax.plot(self.t_series, [t[5] for t in self.joint_position_series])
        ax.plot(self.t_series, [t[6] for t in self.joint_position_series])

        plt.ylabel(r"joint angle $\left[ rad \right]$")
        plt.xlabel(r"$t\left[s\right]$")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
      
        plt.legend([r"".join([r"$\theta_{",str(i),r"}$"]) for i in range(7)],loc='upper left',bbox_to_anchor=(1.05,1))
        if show:
            plt.show()
        else:
            return ax
    
    def plot_theta_dot(self,ax=None,show=False):
        plt.clf()
        ax = plt.gca()
        
        ax.plot(self.t_series, [t[0] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[2] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[3] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[1] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[4] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[5] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[6] for t in self.joint_velocity_series])

        plt.ylabel(r"joint velocity $\left[ \frac{rad}{s} \right]$",fontsize=20)
        plt.xlabel(r"$t\left[s\right]$" ,fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
      
        plt.legend([r"".join([r"$\dot{\theta_{",str(i),r"}}$"]) for i in range(7)],loc='upper left',bbox_to_anchor=(1.05,1))
        if show:
            plt.show()
        else:
            return ax

    def plot_theta_dot_vs_true(self,ax=None,show=False):
        plt.clf()
        ax = plt.gca()
        
        ax.plot(self.t_series, [t[0] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[2] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[3] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[1] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[4] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[5] for t in self.joint_velocity_series])
        ax.plot(self.t_series, [t[6] for t in self.joint_velocity_series])

        ax.plot(self.t_series, [t[0] for t in self.true_joint_velocity_series],'--')
        ax.plot(self.t_series, [t[2] for t in self.true_joint_velocity_series],'--')
        ax.plot(self.t_series, [t[3] for t in self.true_joint_velocity_series],'--')
        ax.plot(self.t_series, [t[1] for t in self.true_joint_velocity_series],'--')
        ax.plot(self.t_series, [t[4] for t in self.true_joint_velocity_series],'--')
        ax.plot(self.t_series, [t[5] for t in self.true_joint_velocity_series],'--')
        ax.plot(self.t_series, [t[6] for t in self.true_joint_velocity_series],'--')

        plt.ylabel(r"joint velocity $\left[ \frac{rad}{s} \right]$",fontsize=20)
        plt.xlabel(r"$t\left[s\right]$" ,fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
      
        plt.legend([r"".join([r"$\dot{\theta_{",str(i),r"}}$"]) for i in range(7)],loc='upper left',bbox_to_anchor=(1.05,1))
        if show:
            plt.show()
        else:
            return ax

    def plot_target_dist(self,ax=None,show=False):
        if not ax:
            plt.clf()
            ax = plt.gca()
        plt.ylabel(r"distance to target $\left[ mm \right]$",fontsize=20)
        plt.xlabel(r"$t\left[s\right]$",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        unit_factor = 1000
        errors = [np.linalg.norm(self.task["p_target"]-p)*unit_factor for p in self.position_series]
        plt.plot(self.t_series,errors)
        if show:
            plt.show()
        else:
            return ax

    def plot_velocity(self,ax=None,show=False):
        if not ax:
            plt.clf()
            ax = plt.gca()
        plt.ylabel(r"end effector speed $\left[ \frac{m}{s} \right]$",fontsize=20)
        plt.xlabel(r"$t\left[s\right]$",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.plot(self.t_series[1:],self.velocity_series)
        if show:
            plt.show()
        else:
            return ax

    def plot_rot_diff(self,ax=None,show=False):
        if not ax:
            plt.clf()
            ax = plt.gca()
        plt.ylabel(r"$pp_target^T$",fontsize=20)
        plt.xlabel(r"$t\left[s\right]$",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.plot(self.t_series[1:],[ np.dot(listify(M[0:3,0:3] @ np.array([[0,1,0]]).T),listify(self.task["R_target"]@np.array([[0,1,0]]).T)) for M in self.SE3_series])
        if show:
            plt.show()
        else:
            return ax

    def plot_obstacle_pole(self,ax=None,show=False):
        if ax == None:
            ax = plt.figure().add_subplot(1,1,1,projection='3d')

        self.plot_box(self.task["p_obst"][0,0],self.task["p_obst"][2,0],self.task["xd_obst"],self.task["zd_obst"],self.task["h_obst"],h_table=self.task["p_obst"][1,0],ax=ax)

        if show:
            plt.show()
        else:
            return ax
    def plot_box(self, xc, zc, xw, zw, h, ax = None, show=False, h_table = 0):
        if ax == None:
            ax = plt.figure().add_subplot(1,1,1,projection='3d')
    
        xmin,xmax = xc-(xw/2),xc+(xw/2)
        zmin,zmax = zc-(zw/2),zc+(zw/2)
        x = [xmin,xmax]
        z = [zmin,zmin+0.0001]
        X, Z = z =np.meshgrid(x, z)
        Y = np.array([[0+h_table,0+h_table],[h+h_table,h+h_table]])

        ax.plot_surface(Z, X, Y, linewidth=1.2, color='lightgray',edgecolors='k',zorder=-1,alpha=0.2)

        x = [xmin,xmin-0.0001]
        z = [zmin,zmax]
        X, Z = z =np.meshgrid(x, z)
        Y = np.array([[0+h_table,h+h_table],[0+h_table,h+h_table]])

        ax.plot_surface(Z, X, Y, linewidth=1.2, color='lightgray',edgecolors='k',zorder=-1,alpha=0.2)

        x = [xmax,xmax+0.0001]
        z = [zmin,zmax]
        X,Z = z =np.meshgrid(x, z)
        Y = np.array([[0+h_table,h+h_table],[0+h_table,h+h_table]])

        ax.plot_surface(Z, X, Y, linewidth=1.2, color='lightgray',edgecolors='k',zorder=-1,alpha=0.2)

        x = [xmin,xmax]
        z = [zmax,zmax+0.0001]
        X, Z = z =np.meshgrid(x, z)
        Y = np.array([[0+h_table,0+h_table],[h+h_table,h+h_table]])

        ax.plot_surface(Z, X, Y, linewidth=1.2, color='lightgray',edgecolors='k',zorder=-1,alpha=0.2)

        x = [xmin,xmax]
        z = [zmin,zmax]
        X, Z = z =np.meshgrid(x, z)
        Y = np.array([[h+h_table,h+h_table],[h+h_table,h+h_table]])

        ax.plot_surface(Z, X, Y, linewidth=1.2, color='lightgray',edgecolors='k',zorder=-1,alpha=0.2)

        if show:
            plt.show()
        else:
            return ax


    def export_plots(self,outpath='outputs',expdir='other',expname='???',**kwargs):
        '''exports a number of plots to a specified directory as encapsulated postscript files'''
        filedir = os.path.join(outpath,expdir)
        try:
            os.mkdir(filedir)
        except:
            pass

        try:
            ax = self.plot_rot_diff()
            ax.figure.savefig(os.path.join(filedir,''.join(['rotplot_',expname,'.pdf'])),bbox_inches='tight')
        except:
            pass

        try:
            ax = self.plot_box(self.task["p_obst"][0,0],self.task["p_obst"][2,0],self.task["xd_obst"],self.task["zd_obst"],self.task["h_obst"],h_table=self.task["p_obst"][1,0])
        except:
            ax = plt.figure().add_subplot(1,1,1,projection='3d')
            print("no obstacle plotted")
        ax = self.plot_arm(ax=ax,i=0)
        ax = self.plot_trajectory(ax=ax)
       
        xmin = min([*(ax.get_xlim()),*(ax.get_ylim()),*(ax.get_zlim())])
        xmax = max([*(ax.get_xlim()),*(ax.get_ylim()),*(ax.get_zlim())])
        ax.set_xlim3d(left=xmin, right=xmax)
        ax.set_ylim3d(bottom=xmin, top=xmax)
        ax.set_zlim3d(bottom=xmin, top=xmax)
        #with open(os.path.join(filedir,''.join(['interactive_armplot_',expname,'.dat'])),'wb') as f:
        #    f.write(dumps(ax.figure))
        ax.figure.savefig(os.path.join(filedir,''.join(['armplot_',expname,'.pdf'])),bbox_inches='tight')

        ax = self.plot_thetas()
        ax.figure.savefig(os.path.join(filedir,''.join(['thetaplot_',expname,'.pdf'])),bbox_inches='tight')
        
        ax = self.plot_theta_dot()
        ax.figure.savefig(os.path.join(filedir,''.join(['thetadotplot_',expname,'.pdf'])),bbox_inches='tight')
        
        ax = self.plot_theta_dot_vs_true()
        ax.figure.savefig(os.path.join(filedir,''.join(['thetadotcmpplot_',expname,'.pdf'])),bbox_inches='tight')

        ax = self.plot_velocity()
        ax.figure.savefig(os.path.join(filedir,''.join(['velocityplot_',expname,'.pdf'])),bbox_inches='tight')

        ax = self.plot_trajectory_components()
        ax.figure.savefig(os.path.join(filedir,''.join(['trajectorycomponentplot_',expname,'.pdf'])),bbox_inches='tight')

        ax = self.plot_curvature()
        ax.figure.savefig(os.path.join(filedir,''.join(['curvatureplot_',expname,'.pdf'])),bbox_inches='tight')

        ax = self.plot_target_dist()
        ax.figure.savefig(os.path.join(filedir,''.join(['target_distplot_',expname,'.pdf'])),bbox_inches='tight')

        ax = self.plot_torsion()
        plt.ylim(0,100)
        ax.figure.savefig(os.path.join(filedir,''.join(['torsionplot_',expname,'.pdf'])),bbox_inches='tight')
    def export_dataset(self,outpath='outputs',expdir='other',expname='???',**kwargs):
        try:
            os.mkdir(os.path.join(outpath,expdir))
        except:
            pass
        print(os.path.join(outpath,expdir,''.join(['dataset_',expname,'.dat'])))
        with open(os.path.join(outpath,expdir,''.join(['dataset_',expname,'.dat'])),'wb') as f:
            f.write(dumps(self.__dict__))

    def import_dataset(self,outpath='outputs',expdir='other',expname='???',filepath=None):
        if not filepath:
            filepath = os.path.join(outpath,expdir,''.join(['dataset_',expname,'.dat']))
        with open(filepath,'rb') as f:
            self.__dict__.update(loads(f.read()))

    def reset(self):
        self.__init__(KC=self.KC,RC = self.RC,task = self.task)


