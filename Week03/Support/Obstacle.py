import numpy as np
from math_functions import *


class Polygon:
	"""
	Obstacles are represented as polygons
	Polygons are defined as an array with n rows (vertices) and 2 columns
	
	"""

	def __init__(self, vertices=np.zeros((4,2))):
		self.vertices = vertices
	
	def compute_distance_point_to_polygon(self, point_q):
		"""
		Compute distance from point_q to the closest point in the polygon
	
		Method returns:
		- dist: minimal distance from point_q to polygon
		- indices of segment closest to point_q

		"""        
		dist = np.inf
		segment_idx = None
		closest_point = None
		len_polygon = self.vertices.shape[0]
		
		for i in range(len_polygon):
			case, seg_dist, _ = compute_distance_point_to_segment(self.vertices[i],
																  self.vertices[(i+1) % len_polygon], point_q)
			if seg_dist <= dist:
				dist = seg_dist
				if case == 0:
					closest_point = i
				elif case == 1:
					closest_point = i
				else:
					closest_point = (i+1) % len_polygon                
		
		segment_idx = (closest_point, (closest_point+1) % len_polygon)
		# print("Closest segment is {}, {}".format(segment_idx[0], segment_idx[1]))
		return dist, segment_idx
			
	def compute_tangent_vector_to_polygon(self, point_q, idx):  
		
		"""
		Determines the unit-length vector tangent at point_q to the polygon
			
		Method returns:
		   tangent vector

		"""  
		   
		v1 = self.vertices[idx[0]]
		v2 = self.vertices[idx[1]]
			
		case, seg_dist, closest_point = compute_distance_point_to_segment(v1, v2, point_q)    
	
		tangent_vector = (v2-v1)/np.linalg.norm(v2-v1)
		
		return tangent_vector
	

	def plot_obstacle(self):
		ox = []
		oy = []

		for i in range(self.vertices.shape[0]):
			x0, y0 = self.vertices[i]
			x1, y1 = self.vertices[(i+1) % self.vertices.shape[0]]

			if x0 == x1:
				start = np.min([y0, y1])
				end = np.max([y0, y1])
				y_values = np.linspace(start, end, int(np.ceil((end-start)/0.2)), endpoint=True) 
				x_values = [x0] * len(y_values)
				ox.extend(x_values)
				oy.extend(y_values)

			if y0 == y1:
				start = np.min([x0, x1])
				end = np.max([x0, x1])
				x_values = np.linspace(start, end, int(np.ceil((end-start)/0.2)), endpoint=True) 
				y_values = [y0] * len(x_values)
				ox.extend(x_values)
				oy.extend(y_values)

		return ox, oy


	def to_display_format(self, screen_height):
		coordinates = [coordinates_to_pygame(v, screen_height) for v in self.vertices[0:-1]]
		return coordinates


	def is_in_collision_with_points(self, points, min_dist=2.5):

		# First check if point is within polygon
		points_in_collision = []

		for point in points:
			count_collisions = 0
			p_x, p_y = point
			for i in range(self.vertices.shape[0]-2):
				j = i - 1 if i != 0 else self.vertices.shape[0]-2
				v1 = self.vertices[i]
				v2 = self.vertices[j]

				if ((v1[1] > p_y) != (v2[1] > p_y)) and (p_x < (v2[0] - v1[0]) * (p_y - v1[1]) / (v2[1] - v1[1] + v1[0])):
					count_collisions += 1

			if count_collisions % 2 == 1:
				points_in_collision.append(point)
		
		if len(points_in_collision):
			return True

		# Second check if point is in collision with edges
		dist, _ = self.compute_distance_point_to_polygon(points[-1])
		if dist < min_dist:
			return True

		return False


	def get_perimeter(self):

		perimeter = 0

		for i in range(self.vertices.shape[0]-1):
			v1 = self.vertices[i]
			v2 = self.vertices[i+1]

			perimeter += compute_distance_between_points(v1, v2)

		return perimeter


class Rectangle(Polygon):
	
	def __init__(self, origin=np.zeros(2), width=100, height=20):
		self.width = width
		self.height = height
		self.origin = origin
				
		v1 = origin
		v2 = origin + np.array([width, 0])
		v3 = origin + np.array([width, -height])
		v4 = origin + np.array([0, -height])
		
		Polygon.__init__(self, vertices=np.array([v1, v2, v3, v4]))
		
	def to_display_format(self, screen_height):
		py_origin = coordinates_to_pygame(self.origin, screen_height)
		return (py_origin[0], py_origin[1], self.width, self.height)

	def plot_obstacle(self):
		return super().plot_obstacle()


class Circle:
	
	def __init__(self, c_x, c_y, radius):
		self.center = np.array([c_x, c_y])
		self.radius = radius

	def is_in_collision_with_points(self, points):

		dist = []
		for point in points:
			dx = self.center[0] - point[0]
			dy = self.center[1] - point[1]

			dist.append(dx * dx + dy * dy)

		if np.min(dist) <= self.radius ** 2:
			return True

		return False  # safe