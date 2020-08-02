import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
import ipywidgets as widgets

class NotebookChecker:
    
    def __init__(self):
        self.x = np.linspace(1, 2 * np.pi, 100)
        fig, ax = plt.subplots()
        self.line, = ax.plot(self.x, np.sin(self.x))
        ax.grid(True)
        self.figure = fig
        
        self.int_slider = widgets.IntSlider(value=0, min=0, max=10, step=1, description='$\omega$',
                                            continuous_update=False)
        self.int_slider.observe(self.update, 'value')
        self.text_secret = widgets.Text(value='', description='', continuous_update=False, disabled=True)
        
        controls = widgets.HBox([self.int_slider, self.text_secret])
        display(controls)
 
    def update(self, change):
        self.line.set_ydata(np.sin(change.new * self.x))
        self.figure.canvas.draw()
        if change.new == 10:
            self.text_secret.value = "Atonal Tweezers"
        else:
            self.text_secret.value = ''
