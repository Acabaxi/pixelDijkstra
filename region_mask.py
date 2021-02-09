import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import medial_axis, skeletonize, binary_dilation

import flood_fill_distance

import piecewise_fit

import skely_prune


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
    """"Removes regions larger than a rectangle of size <size>"""
    elem = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    ay = cv.morphologyEx(img, cv.MORPH_OPEN, elem)
    ay = cv.dilate(ay, elem)
    ay = cv.subtract(img_, ay)

    return ay


def split_line_choose_region(line_skeleton, show_=False, rotated_rect=-1):
    """Rotated_rect: \n
    -1: Upright rectangle \n
    0:  minAreaRect; Rotated rectangle \n
    1:  Convex Hull \n
    2;  Fit ellipse; Doesnt work quite well \n
    """
    line_skeleton_tmp = np.array(line_skeleton, dtype=np.uint8) * 255
    viz_copy = line_skeleton_tmp.copy()

    # Crop region including line skeleton
    all_ones = np.zeros(line_skeleton_tmp.shape, dtype=np.uint8)
    line_skel_crop = line_skeleton_tmp.copy()  # [y:y + h, x + x:w]

    if rotated_rect == -1:
        rect = cv.boundingRect(np.array(line_skeleton_tmp))
        x, y, w, h = rect
        print("Rectangle", rect)
        # all_ones = np.ones(line_skeleton_tmp.shape, dtype=np.uint8) * 255
        all_ones = np.zeros(line_skeleton_tmp.shape, dtype=np.uint8)
        all_ones[y:y + h, x:x + w] = 255

    if rotated_rect == 0:
        # minAreaRect (convexHull) needs list of points, not image
        idx_ = cv.findNonZero(line_skeleton_tmp)
        rect = cv.minAreaRect(idx_)
        box_pts = np.int0(cv.boxPoints(rect))
        print("Box poibnts", box_pts, box_pts)
        cv.drawContours(all_ones, [np.int0(box_pts)], 0, color=255, thickness=cv.FILLED)

    elif rotated_rect == 1:
        cc, hie = cv.findContours(line_skeleton_tmp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hl = []
        for asd in range(len(cc)):
            hull = cv.convexHull(cc[asd])
            hl.append(hull)

        for asd in range(len(cc)):
            cv.drawContours(viz_copy, hl, asd, color=255)

    inv_skel = all_ones - line_skel_crop

    if show_:
        plt.figure()
        plt.title("inv mask")
        plt.imshow(inv_skel)
        plt.show()
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(inv_skel, connectivity=4)
    un = np.unique(labels)

    # Descobrir label da regiao de area maxima
    max_label = max(range(1, stats.shape[0]), key=lambda kk: stats[kk][4])
    pp = np.where(labels == max_label)

    mask_ = np.zeros(line_skeleton.shape, dtype=np.uint8)
    mask_[pp] = 255

    if show_:
        print("Num zones", un)
        print("Zone stats", stats)
        plt.figure()
        plt.title("maks")
        plt.imshow(mask_)
        plt.show()

    return mask_


def process_segmentation(img_, show_=False):
    """"Takes a mask of the segmented line, cleans it and returns clean mask\n
    1. Removes big blobs: the lines we are looking for are thin, not blobs\n
    2. Finds contours\n
    3. Removes contours with small areas\n
    """
    img_ = clean_big_objects(img_, 50)

    if show_:
        plt.figure()
        plt.title("Removed big objects")
        plt.imshow(img_)
        plt.show()

    contours_, hierarchy_ = cv.findContours(img_, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    draw_contour_img = np.zeros(img.shape[:2], dtype=np.uint8)
    filt_contours_ = []
    cont_areas_ = []
    filt_hier_ = []
    for contour_num_ in range(len(contours_)):
        area_ = cv.contourArea(contours_[contour_num_])
        if area_ > 500:
            cv.drawContours(draw_contour_img, contours_, contour_num_, color=255, thickness=cv.FILLED)
            filt_contours_.append(contours_[contour_num_])
            cont_areas_.append(area_)
            # filt_hier.append(hierarchy[cont])

    if show_:
        plt.figure()
        plt.title("Filtered contours")
        plt.imshow(draw_contour_img)
        plt.show()

    return draw_contour_img


def get_mask_from_line_segmentation(img_):
    filtered_image_ = process_segmentation(img_, show_=True)

    skel_img_ = filtered_image_.astype(bool)

    skel_ = skeletonize(skel_img_)
    skel_ = skely_prune.to_graph(skel_, filtered_image_)

    seg_mask_ = split_line_choose_region(skel_, True, rotated_rect=0)

    return seg_mask_


def trash_function_do_not_use():

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
    # contours = [filt_contours[sorted_args[0]], filt_contours[sorted_args[1]]]
    contours = [filt_contours[sorted_args[0]]]
    # hierarchy = filt_hier

    zone_boundary_list = []
    for i in range(len(filt_contours)):
        area = cv.contourArea(filt_contours[i])
        if area < 500:
            continue
        draw_img, pts_list, pts_list_cv = draw_get_contour(dc, filt_contours, i)

        # Skeletonize before approxPolyDP
        # Neater results than with whole shape
        skel_img = draw_img.astype(np.float64) / 255. #Skeletonize needs float
        skel = skeletonize(skel_img)

        #SKelly prune
        skel = skely_prune.to_graph(skel, img)

        skel_pts = np.where(skel == 1)
        skel_pts = list(zip(skel_pts[0], skel_pts[1]))

        split_line_choose_region(skel, True, rotated_rect=0)

        # Find skeleton extremes (Dijkstra)
        ff = flood_fill_distance.FloodFill(skel, skel_pts)
        maps, sorted_points = ff.run(show=True)
        # sort pixels according to their distance on distance map
        skel_pts = sorted(skel_pts, key=lambda x: maps[x])
        skel_pts_cv = []
        # Python libs (skeletonize) return pixels in a (Y,X) order
        for pt in skel_pts:  # Flip pixel order to match (X,Y) of OpenCV
            skel_pts_cv.append(pt[::-1])

        # Skeleton polyline
        draw_img_skel = rgb_draw_img(draw_img)
        polyline_skel = cv.approxPolyDP(np.array(skel_pts_cv), epsilon=10, closed=False)
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



if __name__ == '__main__':
    print('PyCharm')
    img = cv.imread("./line_mask.png", 0)
    print(img.shape)

    seg_mask = get_mask_from_line_segmentation(img)

    plt.figure()
    plt.title("skeleton")
    plt.imshow(seg_mask)
    plt.show()