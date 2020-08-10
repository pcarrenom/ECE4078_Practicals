import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
import ipywidgets as widgets
import threading as thrd
import time

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
class Renderer(thrd.Thread):
    
    #Make singleton
    _instance = None
    
    def __init__(self):
        # Call the Thread class's init function
        thrd.Thread.__init__(self)

        
    def initialize(self, my_robot=None, dt_data=0.02, max_iterations=60):
        self.lock = thrd.Lock()
        
        self.robot_obj = my_robot
        self.initialized = False
        self.paused = False
        self.cur_frame = 0
        self.max_cycles = max_iterations
        self.dt_data = dt_data
                
        # Initialize figure
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set_xlim([-4,-1.5])
        ax.set_ylim([-3.5,-1.5])
        ax.tick_params(axis='both', which='major', labelsize=7)
        plt.title('Overhead View')
        plt.xlabel('X (m)',weight='bold')
        plt.ylabel('Y (m)',weight='bold')

        self.figure = fig
        
        self.line, = ax.plot(self.robot_obj.x, self.robot_obj.y)
            
        # Create Robot Axes 
        self.robot_ax = []
        self.robot_ax.append(FancyArrowPatch((0,0), (0.15,0),
                                            mutation_scale=8,color='red'))
        self.robot_ax.append(FancyArrowPatch((0,0), (0,0.15),
                                            mutation_scale=8,color='green'))
        
        # Apply translation and rotation as specified by current robot state
        cos_theta, sin_theta = np.cos(self.robot_obj.theta), np.sin(self.robot_obj.theta)
        Tw_r = np.eye(3)
        Tw_r[0:2,2] = [self.robot_obj.x, self.robot_obj.y]
        Tw_r[0:2,0:2] = [[cos_theta,-sin_theta],[sin_theta,cos_theta]]
        Tw_r_obj = transforms.Affine2D(Tw_r)
        self.ax_trans = ax.transData
        self.robot_ax[0].set_transform(Tw_r_obj+self.ax_trans)
        self.robot_ax[1].set_transform(self.robot_ax[0].get_transform())
        ax.add_patch(self.robot_ax[0])
        ax.add_patch(self.robot_ax[1])
                
        self.btn_play = widgets.Button(description='Play/Pause', layout=widgets.Layout(flex='1 1 0%', width='auto'), 
                                       button_style='success')
        self.btn_play.on_click(self.pause)
        
        self.btn_reset = widgets.Button(description='Reset', layout=widgets.Layout(flex='1 1 0%', width='auto'), 
                                        button_style='success')

        self.frame_counter = widgets.IntText(description='Frame', layout=widgets.Layout(width='150px', height='80px'), disabled=True)

        self.btn_reset.on_click(self.reset)
              
        controls = widgets.HBox([self.btn_play, self.btn_reset, self.frame_counter])
                    
        display(controls)
        
        if not self.is_alive():
            self.start()
            
        self.initialized = True
                            
    #Render Loop
    def run(self):
        while True:
            if self.paused == False:
                self.robot_obj.drive(self.dt_data)
                
                # Determine frame to plot
                self.cur_frame += 1
                if self.cur_frame >= self.max_cycles:
                    self.cur_frame = 0
                    self.robot_obj.reset()
                if self.initialized == True:
                    self.render()
            time.sleep(0.2)

            
    def render(self):
        self.lock.acquire()
            
        self.figure.canvas.draw_idle()
        
        # Render robot
        (curr_x, curr_y, curr_theta) = self.robot_obj.states[self.cur_frame]
        c, s = np.cos(curr_theta), np.sin(curr_theta)
        Tw_r = np.eye(3)
        Tw_r[0:2,2] = [curr_x, curr_y]
        Tw_r[0:2,0:2] = [[c,-s],[s,c]]
        Tw_r_obj = transforms.Affine2D(Tw_r)
        self.robot_ax[0].set_transform(Tw_r_obj+self.ax_trans)
        self.robot_ax[1].set_transform(self.robot_ax[0].get_transform())
        
        # Render path
        all_states = np.array(self.robot_obj.states)
        self.line.set_data(all_states[1:,0], all_states[1:,1])
        self.frame_counter.value = self.cur_frame
                        
        self.lock.release()
        
    def pause(self,b=None):
        self.paused = not self.paused
        
    def reset(self, b=None):
        self.paused = not self.paused
        self.robot_obj.reset()
        self.cur_frame = 0 
        self.paused = not self.paused
