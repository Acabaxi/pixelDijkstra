import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import medial_axis, skeletonize

import flood_fill_distance
from scipy.optimize import curve_fit
import piecewise_fit


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


def tuple_format(ar_):
    return ar_[0], ar_[1]


def draw_get_contour(img_, contours_, i_):
    draw_img_ = np.zeros(img_.shape[:2], dtype=np.uint8)
    draw_img_ = cv.drawContours(draw_img_, contours_, i_, 255, thickness=cv.FILLED, )
    pts_ = np.where(draw_img_ == 255)
    # pts_nz = cv.findNonZero(draw_img)

    pts_list_ = list(zip(pts_[0], pts_[1]))
    pts_list_cv_ = np.array(list(zip(pts_[1], pts_[0])))

    return draw_img_, pts_list_, pts_list_cv_

if __name__ == '__main__':
    print('PyCharm')
    img = cv.imread("./zone_draw.png", 0)
    print(img.shape)

    contours, hierarchy = cv.findContours(img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    hulls = []
    for i in range(len(contours)):

        # print(np.squeeze(contours[0]))
        hull_points = cv.convexHull(contours[i])
        hulls.append(hull_points)

    zone_boundary_list = []
    for i in range(len(contours)):
        draw_img, pts_list, pts_list_cv = draw_get_contour(img, contours, i)

        # Find contour extremes (Dijkstra)
        ff = flood_fill_distance.FloodFill(draw_img, pts_list)
        maps, sorted_points = ff.run()

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

        # plt.figure()
        # plt.title("Medial")
        # plt.imshow(draw_img_skel)
        # plt.show()

        line_obj = piecewise_fit.LineClass(np.squeeze(polyline_skel))
        zone_boundary_list.append(line_obj)
        # Polyline entire shape
        # Less clean results
        # draw_img_2 = rgb_draw_img(draw_img)
        # polyline = cv.approxPolyDP(np.array(sorted_points_cv), epsilon=10, closed=False)
        # draw_img_2 = cv.drawContours(draw_img_2, contours, i, (0, 0, 255), thickness=cv.FILLED, )
        # draw_img_2 = cv.polylines(draw_img_2, [polyline], isClosed=False, color=(255, 255, 0))
        #
        # plt.figure()
        # plt.title("Approx poly")
        # plt.imshow(draw_img_2)
        # plt.show()

    # zone_boundary_list[1].flip_line()
    # left_limits = [zone_boundary_list[0][0][0], zone_boundary_list[0][-1][0]]
    left_limits = zone_boundary_list[0].get_limits()
    # right_limits = [zone_boundary_list[1][0][0], zone_boundary_list[1][-1][0]]
    right_limits = zone_boundary_list[1].get_limits()

    if check_border_intersect(left_limits[0], right_limits[0], left_limits[1], right_limits[1]):
        print("Flipping")
        zone_boundary_list[0].flip_line()

    zz = np.zeros(img.shape[:2], dtype=np.uint8)
    zz = cv.polylines(zz, [zone_boundary_list[0].get_points()], color=255, isClosed=False)
    zz = cv.polylines(zz, [zone_boundary_list[1].get_points()], color=255, isClosed=False)

    zz = cv.line(zz, tuple_format(zone_boundary_list[0].get_limits()[0]), tuple_format(zone_boundary_list[1].get_limits()[0]), color=255)
    zz = cv.line(zz, tuple_format(zone_boundary_list[0].get_limits()[1]),
                 tuple_format(zone_boundary_list[1].get_limits()[1]), color=255)

    plt.figure()
    plt.imshow(zz)
    plt.show()

