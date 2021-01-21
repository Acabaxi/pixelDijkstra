import numpy as np
import cv2 as cv


class PiecewiseFit:
    def __init__(self, start, end, contour_points, distance_map):
        self.points = contour_points
        self.distance_map = distance_map

        self.edges = [start, end]
        self.edges_distance = [self.distance_map[start], self.distance_map[end]]

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


