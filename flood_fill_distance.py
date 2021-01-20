import numpy as np
import matplotlib.pyplot as plt


class FloodFill:
    def __init__(self, image, points):
        self.image_shape = image.shape[:2]
        self.map = np.full(self.image_shape, -1500, dtype=np.int16)
        self.visited = np.zeros(self.image_shape, dtype=bool)
        self.inside_pts = np.zeros(self.image_shape, dtype=bool)
        for pt in points:
            self.inside_pts[pt] = True
            self.map[pt] = 9999
        self.map[points[0]] = 0
        self.point_q = [points[0]]

    def fill(self, point_tuple, distance):

        if (point_tuple[0] - 1 > 0) & (not self.visited[(point_tuple[0] - 1, point_tuple[1])]) & (self.inside_pts[(point_tuple[0] - 1, point_tuple[1])]):
            if self.map[(point_tuple[0] - 1, point_tuple[1])] > self.map[point_tuple] + 1:
                self.map[(point_tuple[0] - 1, point_tuple[1])] = self.map[point_tuple] + 1
            self.add_to_q((point_tuple[0] - 1, point_tuple[1]))

        if (point_tuple[0] + 1 < self.image_shape[0]) & (not self.visited[(point_tuple[0] + 1, point_tuple[1])]) & (self.inside_pts[(point_tuple[0] + 1, point_tuple[1])]):
            if self.map[(point_tuple[0] + 1, point_tuple[1])] > self.map[point_tuple] + 1:
                self.map[(point_tuple[0] + 1, point_tuple[1])] = self.map[point_tuple] + 1
            self.add_to_q((point_tuple[0] + 1, point_tuple[1]))

        if (point_tuple[1] - 1 > 0) & (not self.visited[(point_tuple[0], point_tuple[1] - 1)]) & (self.inside_pts[(point_tuple[0], point_tuple[1] - 1)]):
            if self.map[(point_tuple[0], point_tuple[1] - 1)] > self.map[point_tuple] + 1:
                self.map[(point_tuple[0], point_tuple[1] - 1)] = self.map[point_tuple] + 1
            self.add_to_q((point_tuple[0], point_tuple[1] - 1))

        if (point_tuple[1] + 1 < self.image_shape[1]) & (not self.visited[(point_tuple[0], point_tuple[1] + 1)]) & (self.inside_pts[(point_tuple[0], point_tuple[1] + 1)]):
            if self.map[(point_tuple[0], point_tuple[1] + 1)] > self.map[point_tuple] + 1:
                self.map[(point_tuple[0], point_tuple[1] + 1)] = self.map[point_tuple] + 1
            self.add_to_q((point_tuple[0], point_tuple[1] + 1))

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
            # plt.figure()
            # plt.imshow(self.map)
            # plt.show()
