import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

import flood_fill_distance


def biggest_contour_line(contour_):
    aaa = cdist(np.squeeze(contour_), np.squeeze(contour_))
    b = np.unravel_index(np.argmax(aaa, axis=None), aaa.shape)
    cv.line(draw_img, (contour_[b[0]][0][0], contour_[b[0]][0][1]),
    (contour_[b[1]][0][0], contour_[b[1]][0][1]), 255)


if __name__ == '__main__':
    print('PyCharm')
    img = cv.imread("./lines.png", 0)
    print(img.shape)

    contours, hierarchy = cv.findContours(img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    hulls = []
    for i in range(len(contours)):

        # print(np.squeeze(contours[0]))
        hull_points = cv.convexHull(contours[i])
        hulls.append(hull_points)

    for i in range(len(contours)):
        draw_img = np.zeros(img.shape[:2], dtype=np.uint8)
        draw_img = cv.drawContours(draw_img, contours, i, 255, thickness=cv.FILLED, )
        pts = np.where(draw_img == 255)
        pts_list = list(zip(pts[0], pts[1]))
        print(pts_list[0])
        ff = flood_fill_distance.FloodFill(draw_img, pts_list)

        ff.cycle_q()
        maps = ff.get_map()
        plt.figure()
        plt.imshow(maps)
        plt.show()
        print("Hull points", len(hulls[i]))
    print(len(contours))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
