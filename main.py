import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import medial_axis, skeletonize

import flood_fill_distance
from scipy.optimize import curve_fit


def rgb_draw_img(img_):
    new_shape = img_.shape[:2] + (3,)
    draw_img_ = np.zeros(new_shape, dtype=np.uint8)
    return draw_img_


def skeletonize_ocv(img_):

    img_copy = img_.copy()
    skel = np.zeros(img_.shape[:2], np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        print("Not done")
        open = cv.morphologyEx(img_copy, cv.MORPH_OPEN, element)
        temp = cv.subtract(img_copy, open)
        eroded = cv.erode(img_copy, element)
        skel = cv.bitwise_or(skel, temp)
        img_copy = eroded.copy()
        done = (cv.countNonZero(img_copy) == 0)
    plt.figure()
    plt.imshow(skel)
    plt.show()


    return skel


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
        pts_list_cv = np.array(list(zip(pts[1], pts[0])))
        print(pts_list[0])

        # Find contour extremes (Dijkstra)
        ff = flood_fill_distance.FloodFill(draw_img, pts_list)
        ff.double_run()
        maps = ff.get_map()
        sorted_points = ff.sort_points()
        sorted_points_cv = []
        for pt in sorted_points:
            sorted_points_cv.append(pt[::-1])

        # Skeletonize before approxPolyDP
        # Neater results than with whole shape
        skel_img = draw_img.astype(np.float64) / 255. # Skeletonize needs float
        skel = skeletonize(skel_img)
        skel_pts = np.where(skel == 1)

        skel_pts = list(zip(skel_pts[0], skel_pts[1]))
        # sort pixels according to their distance on distance map
        skel_pts = sorted(skel_pts, key=lambda x: maps[x])
        skel_pts_cv = []
        # Python libs (skeletonize) return pixels in a (Y,X) order
        for pt in skel_pts: # Flip pixel order to match (X,Y) of OpenCV
            skel_pts_cv.append(pt[::-1])

        # Skeleton polyline
        draw_img_skel = rgb_draw_img(draw_img)
        polyline_skel = cv.approxPolyDP(np.array(skel_pts_cv), epsilon=5, closed=False)
        draw_img_skel = cv.drawContours(draw_img_skel, contours, i, (0, 0, 255), thickness=cv.FILLED, )
        for pt in skel_pts:
            draw_img_skel[pt] = (255, 0, 0)
        draw_img_skel = cv.polylines(draw_img_skel, [polyline_skel], isClosed=False, color=(255, 255, 0))

        plt.figure()
        plt.title("Medial")
        plt.imshow(draw_img_skel)
        plt.show()

        # Polyline entire shape
        # Less clean results
        draw_img_2 = rgb_draw_img(draw_img)
        polyline = cv.approxPolyDP(np.array(sorted_points_cv), epsilon=10, closed=False)
        draw_img_2 = cv.drawContours(draw_img_2, contours, i, (0, 0, 255), thickness=cv.FILLED, )
        draw_img_2 = cv.polylines(draw_img_2, [polyline], isClosed=False, color=(255, 255, 0))

        plt.figure()
        plt.title("Approx poly")
        plt.imshow(draw_img_2)
        plt.show()




        # plt.figure()
        # plt.imshow(maps)
        # plt.show()
        print("Hull points", len(hulls[i]))
    print(len(contours))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
