import numpy as np

class PenguinPi(object):
	
	def __init__(self, wheels_width=0.15, wheels_radius=0.1):
		
		# Robot state
		self.x = 0
		self.y = 0
		self.theta = 0

		# Robot hardware parameters
		self.wheels_width = wheels_width
		self.wheels_radius = wheels_radius
		
		# List of all PenguiPi states in current simulation and wheel speeds
		self.states = []
		
		# Control inputs
		self.linear_velocity = 0
		self.angular_velocity = 0


	def __convert_wheel_speeds__(self, left_speed, right_speed):
		# Convert to m/s
		left_speed_m = left_speed * self.wheels_radius
		right_speed_m = right_speed * self.wheels_radius

		# Compute the linear and angular velocity
		self.linear_velocity = (left_speed_m + right_speed_m) / 2.0
		self.angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
							   

	def drive(self, measurement):
		"""
		 Update the PenguiPi state
		"""        

		# Determine linear and angular velocity from wheels speed
		self.__convert_wheel_speeds__(measurement.left_speed, measurement.right_speed)
		dt = measurement.dt

		# Remember that the PenguiPi current state is given by self.x, self.y, self.theta        
		# Apply the velocities
		if self.angular_velocity == 0:
			# Drive straight
			next_x = self.x - np.cos(self.theta)*self.linear_velocity*dt
			next_y = self.y - np.sin(self.theta)*self.linear_velocity*dt
			next_theta = self.theta
		else:
			# Make a turn
			R = self.linear_velocity / self.angular_velocity
			next_theta = self.theta + self.angular_velocity*dt
			next_x = self.x + R * (-np.sin(self.theta) + np.sin(next_theta))
			next_y = self.y + R * (np.cos(self.theta) - np.cos(next_theta))
	   
		# Make next state our current state
		self.set_state(next_x, next_y, next_theta)        
			
	def get_state(self):
		"""Return the current robot state. The state is in (x,y,theta) format"""
		return np.array([self.x, self.y, self.theta]).reshape((3,1))
	
	def set_state(self,x=0,y=0,theta=0):
		"""Define the new model state"""
		self.x = x
		self.y = y
		self.theta = theta
		self.states.append((x, y, theta))


	def measure(self, markers, idx_list):
		# Markers are 2d landmarks in a 2xn structure where there are n landmarks.
		# The index list tells the function which landmarks to measure in order.
		
		# Construct a 2x2 rotation matrix from the robot angle
		th = self.theta
		
		Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
		robot_xy = np.array([self.x, self.y]).reshape((2,1))

		measurements = []
		for idx in idx_list:
			marker = markers[:,idx:idx+1]
			marker_bff = Rot_theta.T @ (marker - robot_xy)
			measurements.append(marker_bff)

		# Stack the measurements in a 2xm structure.
		markers_bff = np.concatenate(measurements, axis=1)
		return markers_bff

	
	# Derivatives and Covariance
	# --------------------------

	def derivative_drive(self, drive_meas):
		# Compute the differential of drive w.r.t. the robot state
		DFx = np.zeros((3,3))
		DFx[0,0] = 1
		DFx[1,1] = 1
		DFx[2,2] = 1

		lin_vel = self.linear_velocity
		ang_vel = self.angular_velocity

		self.__convert_wheel_speeds__(drive_meas.left_speed, drive_meas.right_speed)

		dt = drive_meas.dt
		th = self.theta
		if ang_vel == 0:
			DFx[0,2] = -np.sin(th) * self.linear_velocity * dt
			DFx[1,2] = np.cos(th) * self.linear_velocity * dt
		else:
			DFx[0,2] = self.linear_velocity / self.angular_velocity * (np.cos(th+dt*self.angular_velocity) - np.cos(th))
			DFx[1,2] = self.linear_velocity / self.angular_velocity * (np.sin(th+dt*self.angular_velocity) - np.sin(th))

		self.linear_velocity = lin_vel
		self.angular_velocity = ang_vel

		return DFx


	def derivative_measure(self, markers, idx_list):
		# Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
		n = 2*len(idx_list)
		m = 3 + 2*markers.shape[1]

		DH = np.zeros((n,m))

		robot_xy = np.array([self.x, self.y]).reshape((2,1))

		th = self.theta
		Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
		DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

		for i in range(n//2):
			j = idx_list[i]
			# i identifies which measurement to differentiate.
			# j identifies the marker that i corresponds to.

			lmj_inertial = markers[:,j:j+1]
			# lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)

			# robot xy DH
			DH[2*i:2*i+2,0:2] = - Rot_theta.T
			# robot theta DH
			DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
			# lm xy DH
			DH[2*i:2*i+2, 3+2*j:3+2*j+2] = Rot_theta.T
			
		return DH
	

	def covariance_drive(self, drive_meas):
		# Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed
		Jac1 = np.array([[self.wheels_radius/2, self.wheels_radius/2],
				[-self.wheels_radius/self.wheels_width, self.wheels_radius/self.wheels_width]])

		lin_vel = self.linear_velocity
		ang_vel = self.angular_velocity
		
		self.__convert_wheel_speeds__(drive_meas.left_speed, drive_meas.right_speed)
		th = self.theta
		dt = drive_meas.dt
		th2 = th + dt*self.angular_velocity

		# Derivative of x,y,theta w.r.t. lin_vel, ang_vel
		Jac2 = np.zeros((3,2))
		if self.angular_velocity == 0:
			Jac2[0,0] = dt*np.cos(th)
			Jac2[0,1] = dt*np.sin(th)
		else:
			Jac2[0,0] = 1/self.angular_velocity * (np.sin(th2) - np.sin(th))
			Jac2[0,1] = -1/(self.angular_velocity**2) * (np.sin(th2) - np.sin(th)) + \
							self.linear_velocity / self.angular_velocity * (dt * np.cos(th2))

			Jac2[1,0] = -1/self.angular_velocity * (np.cos(th2) - np.cos(th))
			Jac2[1,1] = 1/(self.angular_velocity**2) * (np.cos(th2) - np.cos(th)) + \
							-self.linear_velocity / self.angular_velocity * (-dt * np.sin(th2))
		Jac2[2,1] = dt

		# Derivative of x,y,theta w.r.t. left_speed, right_speed
		Jac = Jac2 @ Jac1

		# Compute covariance
		cov = np.diag((drive_meas.left_cov, drive_meas.right_cov))
		cov = Jac @ cov @ Jac.T

		self.linear_velocity = lin_vel
		self.angular_velocity = ang_vel
		
		return cov