import numpy as np
import cv2 as cv


class LineClass:
    def __init__(self, points):
        self.points = points
        self.limits = [points[0], points[1]]
        # self.distance_map = distance_map

        # self.edges = [start, end]
        # self.edges_distance = [self.distance_map[start], self.distance_map[end]]

    def flip_line(self):
        self.points = self.points[::-1]
        self.limits = self.limits[::-1]

    def get_limits(self):
        return self.points[0], self.points[-1]

    def get_points(self):
        return self.points

    def fit_line_between_edges(self, start_edge, end_edge):
        # Given start and end pixels, get equation
        return

    def maximum_point_distance(self):
        # Calculate error between given line and shape points

        # Filter shape points using distance map
        return

    def point_to_line_distance(self):
        # Get error from a single point to a given line
        return

    def add_edge(self):
        # Add edge to piecewise shape
        return


