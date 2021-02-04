import numpy as np
import matplotlib.pyplot as plt
import math


def is_between(value, minimum, maximum):
    if value >= minimum:
        if value < maximum:
            return True

    return False


class FloodFill:
    def __init__(self, image, points):
        self.image_shape = image.shape[:2]
        self.points = points
        self.map = np.full(self.image_shape, -1, dtype=np.float64)
        self.visited = np.zeros(self.image_shape, dtype=bool)
        self.inside_pts = np.zeros(self.image_shape, dtype=bool)
        for pt in points:
            self.inside_pts[pt] = True
            self.map[pt] = math.inf
        self.map[points[0]] = 0
        self.point_q = [points[0]]

    def fill(self, point_tuple, distance):
        to_visit = ((0, -1), (0, 1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))

        for x_offset, y_offset in to_visit:
            new_point_x = point_tuple[0] + x_offset
            new_point_y = point_tuple[1] + y_offset

            if is_between(new_point_x, 0, self.image_shape[0]) and is_between(new_point_y, 0, self.image_shape[1]):
                if (not self.visited[(new_point_x, new_point_y)]) & (
                self.inside_pts[(new_point_x, new_point_y)]):
                    if self.map[(new_point_x, new_point_y)] > self.map[point_tuple] + 1:
                        self.map[(new_point_x, new_point_y)] = self.map[point_tuple] + 1
                    self.add_to_q((new_point_x, new_point_y))
        #
        # if point_tuple[0] - 1 > 0:
        #     if (not self.visited[(point_tuple[0] - 1, point_tuple[1])]) & (self.inside_pts[(point_tuple[0] - 1, point_tuple[1])]):
        #         if self.map[(point_tuple[0] - 1, point_tuple[1])] > self.map[point_tuple] + 1:
        #             self.map[(point_tuple[0] - 1, point_tuple[1])] = self.map[point_tuple] + 1
        #         self.add_to_q((point_tuple[0] - 1, point_tuple[1]))
        #
        # if point_tuple[0] + 1 < self.image_shape[0]:
        #     if (not self.visited[(point_tuple[0] + 1, point_tuple[1])]) & (self.inside_pts[(point_tuple[0] + 1, point_tuple[1])]):
        #         if self.map[(point_tuple[0] + 1, point_tuple[1])] > self.map[point_tuple] + 1:
        #             self.map[(point_tuple[0] + 1, point_tuple[1])] = self.map[point_tuple] + 1
        #         self.add_to_q((point_tuple[0] + 1, point_tuple[1]))
        #
        # if point_tuple[1] - 1 > 0:
        #     if (not self.visited[(point_tuple[0], point_tuple[1] - 1)]) & (self.inside_pts[(point_tuple[0], point_tuple[1] - 1)]):
        #         if self.map[(point_tuple[0], point_tuple[1] - 1)] > self.map[point_tuple] + 1:
        #             self.map[(point_tuple[0], point_tuple[1] - 1)] = self.map[point_tuple] + 1
        #         self.add_to_q((point_tuple[0], point_tuple[1] - 1))
        #
        # if point_tuple[1] + 1 < self.image_shape[1]:
        #     if (not self.visited[(point_tuple[0], point_tuple[1] + 1)]) & (self.inside_pts[(point_tuple[0], point_tuple[1] + 1)]):
        #         if self.map[(point_tuple[0], point_tuple[1] + 1)] > self.map[point_tuple] + 1:
        #             self.map[(point_tuple[0], point_tuple[1] + 1)] = self.map[point_tuple] + 1
        #         self.add_to_q((point_tuple[0], point_tuple[1] + 1))

        self.visited[point_tuple] = True

    def get_map(self):
        return self.map

    def add_to_q(self, pt_tuple):
        if pt_tuple not in self.point_q:
            self.point_q.append(pt_tuple)

    def cycle_q(self):
        while len(self.point_q) > 0:
            curr_pt = self.point_q.pop(0)
            self.fill(curr_pt, 0)

        idx = np.unravel_index(np.argmax(self.map, axis=None), self.map.shape)
        return idx

    def reset_structures(self):
        self.map = np.full(self.image_shape, -1, dtype=np.float64)
        self.visited = np.zeros(self.image_shape, dtype=bool)
        for pt in self.points:
            self.map[pt] = math.inf

    def double_run(self):
        # Run algorithm on random point on the shape
        max_first_run = self.cycle_q()

        # Reset distance map
        self.reset_structures()

        self.map[max_first_run] = 0
        self.point_q = [max_first_run]

        max_second_run = self.cycle_q()

        return max_first_run, max_second_run

    def sort_points(self):
        sorted_list = sorted(self.points, key=lambda x: self.map[x])
        return sorted_list

    def run(self, show = False):
        self.double_run()
        maps = self.get_map()
        sorted_pts = self.sort_points()

        if show:
            plt.figure()
            plt.title("Distance map")
            plt.imshow(maps)
            plt.show()

        return maps, sorted_pts
