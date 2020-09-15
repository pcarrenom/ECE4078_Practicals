import numpy as np

class PenguinPi(object):

    """Implementation of a differential drive robot

    """
    
    def __init__(self, init_state=np.zeros(3), max_v=10, max_omega=np.pi):

        """
        Initialize a new PenguinPi robot
        :param init_state: Initial state of the robot
        :param max_omega: Maximum angular velocity that can applied to the robot
        :param max_v: Maximum linear velocity that can applied to the robot
        """
        
        # Robot state
        self.x = init_state[0]
        self.y = init_state[1]
        self.theta = init_state[2]
                        
        # Control input bounds
        self.max_linear_velocity = max_v
        self.max_angular_velocity = max_omega
                                 
    def drive(self, v=0, omega=0, dt=0.02):
        """
        Update the PenguiPi state
        :param v: Linear velocity (m/s)
        :param omega: Angular velocity (radians/s)
        :param dt: Delta time, i.e., time elapse since last state update
        """        
        
        # Set control signals within admisible bounds
        v = np.clip(v, -self.max_linear_velocity, self.max_linear_velocity)
        omeage = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)


        if omega == 0:
            next_x = self.x - np.cos(self.theta)*v*dt
            next_y = self.y - np.sin(self.theta)*v*dt
            next_theta = self.theta
        else:
            R = v / omega
            next_theta = self.theta + omega*dt
            next_x = self.x + R * (-np.sin(self.theta) + np.sin(next_theta))
            next_y = self.y + R * (np.cos(self.theta) - np.cos(next_theta))
       
        # Make next state our current state
        self.set_state(next_x, next_y, next_theta)        
        
           
    def reset(self):
        """
        Set robot state back to zero
        """
        self.x, self.y, self.theta = 0, 0, 0
    
    def get_state(self):
        """Return the current robot state. The state is in (x,y,theta) format"""
        return np.array([self.x, self.y, self.theta])
    
    def set_state(self,x=0,y=0,theta=0):
        """Define the new model state"""
        self.x = x
        self.y = y
        self.theta = theta