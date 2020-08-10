import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
import ipywidgets as widgets
import threading as thrd
import time
from matplotlib.lines import Line2D

class Singleton:
    def __init__(self, cls):
        self._cls = cls

    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)

@Singleton
class TestingRenderer(thrd.Thread):
    
    #Make singleton
    _instance = None
    
    def __init__(self):
        # Call the Thread class's init function
        thrd.Thread.__init__(self)

        
    def initialize(self, states, dt_render=0.2, dt_data=0.02):
        self.lock = thrd.Lock()
        
        self.initialized = False
        self.paused = False
        self.cur_frame = 0
        self.dt_render = dt_render
        self.dt_data = dt_data
                        
        # Initialize figure
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.tick_params(axis='both', which='major', labelsize=7)
        plt.title('Overhead View')
        plt.xlabel('X (m)',weight='bold')
        plt.ylabel('Y (m)',weight='bold')

        self.states = states
        self.figure = fig
        
        self.line, = ax.plot(states[0,0], states[0,1])
            
        # Create Robot Axes 
        self.robot_ax = []
        self.robot_ax.append(FancyArrowPatch((0,0), (1,0),
                                            mutation_scale=8,color='red'))
        self.robot_ax.append(FancyArrowPatch((0,0), (0,1),
                                            mutation_scale=8,color='green'))
        
        # Apply translation and rotation as specified by current robot state
        cos_theta, sin_theta = np.cos(states[0,2]), np.sin(states[0,2])
        Tw_r = np.eye(3)
        Tw_r[0:2,2] = [0, 0]
        Tw_r[0:2,0:2] = [[cos_theta,-sin_theta],[sin_theta,cos_theta]]
        Tw_r_obj = transforms.Affine2D(Tw_r)
        self.ax_trans = ax.transData
        self.robot_ax[0].set_transform(Tw_r_obj+self.ax_trans)
        self.robot_ax[1].set_transform(self.robot_ax[0].get_transform())
        ax.add_patch(self.robot_ax[0])
        ax.add_patch(self.robot_ax[1])
                
        
        if not self.is_alive():
            self.start()
            
        self.initialized = True
                            
    #Render Loop
    def run(self):
        while True:
            if self.paused == False:
                self.cur_frame = int(self.cur_frame + self.dt_render/self.dt_data)
                if self.cur_frame >= self.states.shape[0]:
                    self.cur_frame = 0
                if self.initialized == True:
                    self.render()
            time.sleep(self.dt_render)

            
    def render(self):
        self.lock.acquire()
            
        self.figure.canvas.draw_idle()
        
        # Render robot
        curr_state = self.states[self.cur_frame]
        c, s = np.cos(curr_state[2]), np.sin(curr_state[2])
        Tw_r = np.eye(3)
        Tw_r[0:2,2] = curr_state[0:2]
        Tw_r[0:2,0:2] = [[c,-s],[s,c]]
        Tw_r_obj = transforms.Affine2D(Tw_r)
        self.robot_ax[0].set_transform(Tw_r_obj+self.ax_trans)
        self.robot_ax[1].set_transform(self.robot_ax[0].get_transform())
        
        # Render path
        all_states = self.states[:self.cur_frame,:]
        self.line.set_data(all_states[:,0], all_states[:,1])                        
        self.lock.release()
        
    

def display_bicycle_wheels(rear_wheel, front_wheel, theta):               
    # Initialize figure
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_xlim([0,4])
    ax.set_ylim([0,4])
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.title('Overhead View')
    plt.xlabel('X (m)',weight='bold')
    plt.ylabel('Y (m)',weight='bold')

    ax.plot(0,0)
  
    rear_wheel_x = FancyArrowPatch((0,0), (0.4,0),
                                        mutation_scale=8,color='red')
    rear_wheel_y = FancyArrowPatch((0,0), (0,0.4),
                                        mutation_scale=8,color='red')

    front_wheel_x = FancyArrowPatch((0,0), (0.4,0),
                                        mutation_scale=8,color='blue') 
    front_wheel_y = FancyArrowPatch((0,0), (0,0.4),
                                        mutation_scale=8,color='blue')

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]
    
    # Apply translation and rotation as specified by current robot state
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    Tw_rear = np.eye(3)
    Tw_rear[0:2,2] = rear_wheel
    Tw_rear[0:2,0:2] = [[cos_theta,-sin_theta],[sin_theta,cos_theta]]
    Tw_rear_obj = transforms.Affine2D(Tw_rear)

    Tw_front = np.eye(3)
    Tw_front[0:2,2] = front_wheel
    Tw_front[0:2,0:2] = [[cos_theta,-sin_theta],[sin_theta,cos_theta]]
    Tw_front_obj = transforms.Affine2D(Tw_front)

    ax_trans = ax.transData
    
    rear_wheel_x.set_transform(Tw_rear_obj+ax_trans)
    rear_wheel_y.set_transform(rear_wheel_x.get_transform())
    ax.add_patch(rear_wheel_x)
    ax.add_patch(rear_wheel_y)

    front_wheel_x.set_transform(Tw_front_obj+ax_trans)
    front_wheel_y.set_transform(front_wheel_x.get_transform())
    ax.add_patch(front_wheel_x)
    ax.add_patch(front_wheel_y)

    ax.legend(custom_lines, ['Rear Wheel', 'Front Wheel']) 
        