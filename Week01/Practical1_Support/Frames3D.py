import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms

class Frame3D:
    
    def __init__(self):
                
        origin = np.array([0, 0, 0])
        x_axis = np.array([2, 0, 0])
        y_axis = np.array([0, 2, 0])
        z_axis = np.array([0, 0, 2])
 
        self.initial_state = [origin, x_axis, y_axis, z_axis]
            
        # Initialize figure
        fig = plt.figure(figsize=(5, 5))
        fig.canvas.toolbar_position = 'top'
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-3,3),ax.set_ylim(-3,3),ax.set_zlim(-3,3)
        ax.set_xlabel('x'),ax.set_ylabel('y'),ax.set_zlabel('z')

        self.figure = fig

        # Create Robot Axes 
        x_arrow, = ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]],
                           color='r', lw=2, linestyle='dotted')
        y_arrow, = ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]],
                           color='g', lw=2, linestyle='dotted')
        z_arrow, = ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]],
                           color='b', lw=2, linestyle='dotted')
        self.world_ax = [x_arrow, y_arrow, z_arrow]

                    
        # Create Robot Axes 
        x_arrow, = ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]],
                           color='r', lw=3)
        y_arrow, = ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]],
                           color='g', lw=3)
        z_arrow, = ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]],
                           color='b', lw=3)
        self.robot_ax = [x_arrow, y_arrow, z_arrow]
               
    
    def rotate_frame(self, new_rotation):
    	self.figure.canvas.draw_idle()
    	new_o = new_rotation.dot(self.initial_state[0])
    	new_x = new_rotation.dot(self.initial_state[1])
    	new_y = new_rotation.dot(self.initial_state[2])
    	new_z = new_rotation.dot(self.initial_state[3])
    	self.robot_ax[0].set_data_3d([new_o[0], new_x[0]], [new_o[1], new_x[1]], [new_o[2], new_x[2]])
    	self.robot_ax[1].set_data_3d([new_o[0], new_y[0]], [new_o[1], new_y[1]], [new_o[2], new_y[2]])
    	self.robot_ax[2].set_data_3d([new_o[0], new_z[0]], [new_o[1], new_z[1]], [new_o[2], new_z[2]])

    
    def apply_transform(self, new_transform):

    	def to_homogeneous(v):
    		return np.transpose(np.hstack([v, 1]))

    	self.figure.canvas.draw_idle()
    	
    	# Compute new robot frame
    	new_o = np.matmul(new_transform, to_homogeneous(self.initial_state[0]))
    	new_x = np.matmul(new_transform, to_homogeneous(self.initial_state[1]))
    	new_y = np.matmul(new_transform, to_homogeneous(self.initial_state[2]))
    	new_z = np.matmul(new_transform, to_homogeneous(self.initial_state[3]))
    	self.robot_ax[0].set_data_3d([new_o[0], new_x[0]], [new_o[1], new_x[1]], [new_o[2], new_x[2]])
    	self.robot_ax[1].set_data_3d([new_o[0], new_y[0]], [new_o[1], new_y[1]], [new_o[2], new_y[2]])
    	self.robot_ax[2].set_data_3d([new_o[0], new_z[0]], [new_o[1], new_z[1]], [new_o[2], new_z[2]])