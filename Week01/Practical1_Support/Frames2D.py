import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
import ipywidgets as widgets

class Rotation2D:
    
    def __init__(self):
        
        # Initialize figure
        fig = plt.figure(figsize=(5, 5))
        fig.canvas.toolbar_position = 'top'
        ax = plt.gca()
        ax.set_xlim([-2.5,2.5])
        ax.set_ylim([-2.5,2.5])
        _ = ax.set_xticks(np.arange(-2.5, 2.5, 0.5))
        _ = ax.set_yticks(np.arange(-2.5, 2.5, 0.5))
        ax.tick_params(axis='both', which='major', labelsize=7)
        plt.title('Overhead View')
        plt.xlabel('X (m)',weight='bold')
        plt.ylabel('Y (m)',weight='bold')

        self.figure = fig
        
        ax.plot(0,0)

        # Create World Axes
        self.world_ax = []
        self.world_ax.append(FancyArrowPatch((0,0), (0.5,0),
                                            mutation_scale=8,color='blue'))
        self.world_ax.append(FancyArrowPatch((0,0), (0,0.5),
                                            mutation_scale=8,color='blue'))
        
        # Create Robot Axes 
        self.robot_ax = []
        self.robot_ax.append(FancyArrowPatch((0,0), (0.6,0),
                                            mutation_scale=8,color='red'))
        self.robot_ax.append(FancyArrowPatch((0,0), (0,0.6),
                                            mutation_scale=8,color='green'))

        ax.add_patch(self.robot_ax[0])
        ax.add_patch(self.robot_ax[1])
        self.ax_trans = ax.transData
        
        # Add arrows to plot
        ax.add_patch(self.robot_ax[0])
        ax.add_patch(self.robot_ax[1])
        ax.add_patch(self.world_ax[0])
        ax.add_patch(self.world_ax[1])
        
    
    def update_frame(self, new_transforms):
        # transform is a dictionnary that contains a translation and a rotation change to apply to the frame
        # {'translation': np.array (2D), 'rotation': np.array (2x2 matrix)}
        self.figure.canvas.draw_idle()
        trans = new_transforms['translation'] if 'translation' in new_transforms else np.zeros(2)
        rot = new_transforms['rotation'] if 'rotation' in new_transforms else np.eye(2)
        Tw_r = np.eye(3)
        Tw_r[0:2,2] = trans
        Tw_r[0:2,0:2] = rot
        Tw_r_obj = transforms.Affine2D(Tw_r)
        self.robot_ax[0].set_transform(Tw_r_obj+self.ax_trans)
        self.robot_ax[1].set_transform(self.robot_ax[0].get_transform())
        
        
    def reset_frame(self, b=None):
        self.figure.canvas.draw_idle()
        Tw_r = np.eye(3)
        Tw_r_obj = transforms.Affine2D(Tw_r)
        self.robot_ax[0].set_transform(Tw_r_obj+self.ax_trans)
        self.robot_ax[1].set_transform(self.robot_ax[0].get_transform())

