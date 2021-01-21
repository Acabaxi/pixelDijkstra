import numpy as np
import cv2 as cv


def tuple_format(ar_):
    return ar_[0], ar_[1]


def linearize_y_mx_b(endpoints):
    if endpoints[0][0] == endpoints[1][0]:
        m = 99999
    else:
        m = (endpoints[1][1] - endpoints[0][1]) / (endpoints[1][0] - endpoints[0][0])
        b = endpoints[0][1] - m * endpoints[0][0]

    return m, b


def check_border_intersect(bl, br, tl, tr):
    if bl[0] != tl[0]:
        print(bl, br, tl, tr)
        A1 = (bl[1] - br[1]) / (bl[0] - br[0])
        A2 = (tl[1] - tr[1]) / (tl[0] - tr[0])
        b1 = bl[1] - A1 * bl[0]
        b2 = tl[1] - A2 * tl[0]

        Xa = (b2 - b1) / (A1 - A2)
        Ya = A1 * Xa + b1

        if ((Xa < max(min(bl[0], br[0]), min(tl[0], tr[0]))) or
                (Xa > min(max(bl[0], br[0]), max(tl[0], tr[0])))):
            t = 1
        else:
            print("Flip")
            tmp = tr
            tr = tl
            tl = tmp
            return True

    return False


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


class LineRegion:
    def __init__(self, boundary_lines):
        self.boundary_lines = boundary_lines
        self.boundary_polyline = []

    def create_boundary_polyline(self):

        left_limits = self.boundary_lines[0].get_limits()
        # right_limits = [zone_boundary_list[1][0][0], zone_boundary_list[1][-1][0]]
        right_limits = self.boundary_lines[1].get_limits()

        # Special case!
        if not check_border_intersect(left_limits[0], right_limits[0], left_limits[1], right_limits[1]):
            print("Flipping")
            self.boundary_lines[0].flip_line()

        self.boundary_polyline = np.vstack([self.boundary_lines[0].get_points(), self.boundary_lines[1].get_points()])

    def draw_boundary_polyline(self, img_shape):
        if len(self.boundary_polyline) == 0:
            self.create_boundary_polyline()

        if len(img_shape) == 3:
            shape_ = img_shape[:2]
        else:
            shape_ = img_shape

        zz = np.zeros(shape_, dtype=np.uint8)
        zz = cv.polylines(zz, [self.boundary_polyline], color=255, isClosed=True)
        return zz
        # zz = cv.polylines(zz, [zone_boundary_list[0].get_points()], color=255, isClosed=False)
        # zz = cv.polylines(zz, [zone_boundary_list[1].get_points()], color=255, isClosed=False)
        #
        # zz = cv.line(zz, tuple_format(zone_boundary_list[0].get_limits()[0]),
        #              tuple_format(zone_boundary_list[1].get_limits()[0]), color=255)
        # zz = cv.line(zz, tuple_format(zone_boundary_list[0].get_limits()[1]),
        #              tuple_format(zone_boundary_list[1].get_limits()[1]), color=255)
