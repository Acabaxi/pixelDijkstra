import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import medial_axis, skeletonize

import flood_fill_distance
from scipy.optimize import curve_fit
import piecewise_fit

import skely_prune
from skan import draw



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


def draw_get_contour(img_, contours_, i_):
    draw_img_ = np.zeros(img_.shape[:2], dtype=np.uint8)
    draw_img_ = cv.drawContours(draw_img_, contours_, i_, 255, thickness=cv.FILLED, )
    pts_ = np.where(draw_img_ == 255)
    # pts_nz = cv.findNonZero(draw_img)

    pts_list_ = list(zip(pts_[0], pts_[1]))
    pts_list_cv_ = np.array(list(zip(pts_[1], pts_[0])))

    return draw_img_, pts_list_, pts_list_cv_


def clean_big_objects(img_, size):
    elem = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    ay = cv.morphologyEx(img, cv.MORPH_OPEN, elem)
    ay = cv.dilate(ay, elem)
    ay = cv.subtract(img_, ay)

    return ay


if __name__ == '__main__':
    print('PyCharm')
    img = cv.imread("./IMG_7484_mask.png", 0)
    print(img.shape)

    img = clean_big_objects(img, 50)

    plt.figure()
    plt.imshow(img)
    plt.show()
    elem = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    ay = cv.morphologyEx(img, cv.MORPH_OPEN, elem)

    contours, hierarchy = cv.findContours(img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    dc = np.zeros(img.shape[:2], dtype=np.uint8)
    filt_contours = []
    cont_areas = []
    filt_hier = []
    for cont in range(len(contours)):
        area = cv.contourArea(contours[cont])
        if area > 500:
            dc = cv.drawContours(dc, contours, cont, color=255)

            filt_contours.append(contours[cont])
            cont_areas.append(area)
            # filt_hier.append(hierarchy[cont])

    sorted_args = np.argsort(cont_areas)[::-1]

    for a in sorted_args:
        print(cont_areas[a])
    contours = [filt_contours[sorted_args[0]], filt_contours[sorted_args[1]]]
    # hierarchy = filt_hier
    plt.figure()
    plt.imshow(dc)
    plt.show()

    hulls = []
    # for i in range(len(contours)):
    #
    #     # print(np.squeeze(contours[0]))
    #     hull_points = cv.convexHull(contours[i])
    #     hulls.append(hull_points)

    zone_boundary_list = []
    for i in range(len(contours)):
        draw_img, pts_list, pts_list_cv = draw_get_contour(img, contours, i)
        print(len(pts_list))
        print(pts_list_cv.shape)


        # Find contour extremes (Dijkstra)
        ff = flood_fill_distance.FloodFill(draw_img, pts_list)
        maps, sorted_points = ff.run()
        #
        # sorted_points_cv = []
        # for pt in sorted_points:
        #     sorted_points_cv.append(pt[::-1])

        # Skeletonize before approxPolyDP
        # Neater results than with whole shape
        skel_img = draw_img.astype(np.float64) / 255. #Skeletonize needs float
        skel = skeletonize(skel_img)
        skel_pts = np.where(skel == 1)

        skel_pts = list(zip(skel_pts[0], skel_pts[1]))
        # sort pixels according to their distance on distance map
        # skel_pts = sorted(skel_pts, key=lambda x: maps[x])
        skel_pts_cv = []
        # Python libs (skeletonize) return pixels in a (Y,X) order
        for pt in skel_pts: # Flip pixel order to match (X,Y) of OpenCV
            skel_pts_cv.append(pt[::-1])

        #SKelly prune

        skel = skely_prune.to_graph(skel, img)
        skel_pts = np.where(skel == 1)

        skel_pts = list(zip(skel_pts[0], skel_pts[1]))
        # sort pixels according to their distance on distance map
        skel_pts = sorted(skel_pts, key=lambda x: maps[x])
        skel_pts_cv = []
        # Python libs (skeletonize) return pixels in a (Y,X) order
        for pt in skel_pts:  # Flip pixel order to match (X,Y) of OpenCV
            skel_pts_cv.append(pt[::-1])

        # Skeleton polyline
        draw_img_skel = rgb_draw_img(draw_img)
        polyline_skel = cv.approxPolyDP(np.array(skel_pts_cv), epsilon=20, closed=False)
        draw_img_skel = cv.drawContours(draw_img_skel, contours, i, (0, 0, 255), thickness=cv.FILLED, )
        for pt in skel_pts:
            draw_img_skel[pt] = (255, 0, 0)
        draw_img_skel = cv.polylines(draw_img_skel, [polyline_skel], isClosed=False, color=(255, 255, 0))

        plt.figure()
        plt.title("Medial")
        plt.imshow(draw_img_skel)
        plt.show()

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

    region_obj = piecewise_fit.LineRegion(zone_boundary_list)
    aa = region_obj.draw_boundary_polyline(img.shape, isClosed_=True)

    sub_line = zone_boundary_list[0].get_points()[:2]
    sub_line_2 = zone_boundary_list[1].get_points()[-2:]
    xa, ya, m1, b1, m2, b2 = piecewise_fit.intersect_point(sub_line, sub_line_2)
    print(xa, ya, m1, b1, m2, b2)

    ####
    line_y = lambda m, x, b: m * x + b
    line_x = lambda m, y, b: (y - b) / m

    if xa < 0 or xa > img.shape[1] or ya < 0 or ya > img.shape[0]:
        intersect_0 = line_x(m1, 0, b1)
        intersect_max = line_x(m1, img.shape[0], b1)
        aa[0:0 + 5, int(intersect_0 - 5):int(intersect_0 + 5)] = 255

        intersect_0 = line_x(m2, 0, b2)
        intersect_max = line_x(m2, img.shape[0], b2)
        aa[0:0 + 5, int(intersect_0 - 5):int(intersect_0 + 5)] = 255

        print(intersect_0, intersect_max)

    print(sub_line[0]-5, sub_line[0]+5)
    aa[sub_line[0][1]-5:sub_line[0][1]+5, sub_line[0][0]-5:sub_line[0][0]+5] = 255
    aa[sub_line[1][1] - 5:sub_line[1][1] + 5, sub_line[1][0] - 5:sub_line[1][0] + 5] = 255
    aa[sub_line_2[0][1] - 5:sub_line_2[0][1] + 5, sub_line_2[0][0] - 5:sub_line_2[0][0] + 5] = 255
    aa[sub_line_2[1][1] - 5:sub_line_2[1][1] + 5, sub_line_2[1][0] - 5:sub_line_2[1][0] + 5] = 255

    plt.figure()
    plt.imshow(aa)
    plt.show()

    # piecewise_fit.intersect_point()
    # plt.figure()
    # plt.imshow(zz)
    # plt.show()


