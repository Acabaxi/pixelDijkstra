from skan import skeleton_to_csgraph, summarize, Skeleton, branch_statistics, draw
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology


def check_increment_dict(dict_, branch_, key_):
    if key_ in dict_:
        dict_[key_].append(branch_)
    else:
        dict_[key_] = [branch_]


def to_graph(skeletony, img_):
    print("Creating graph", skeletony.dtype)

    w, h = skeletony.shape

    new_img = np.empty((w, h, 3), dtype=np.uint8)
    new_img[:, :, 0] = img_.astype(np.uint8) * 255
    new_img[:, :, 1] = img_.astype(np.uint8) * 255
    new_img[:, :, 2] = img_.astype(np.uint8) * 0

    previous_one_branches = 9999999

    for i in range(7):

        skeleton_obj = Skeleton(skeletony, source_image=new_img, keep_images=True, unique_junctions=True)

        # https://github.com/jni/skan/issues/92
        branch_data = summarize(skeleton_obj)

        thres = 100

        bt = branch_data['branch-type'].value_counts()

        if 1 in bt.keys():
            num_ones = bt[1]
            if num_ones < previous_one_branches:
                previous_one_branches = bt[1]
            elif num_ones == previous_one_branches:
                print("Cleaned all 1 branches")
                break
        else:
            print("No 1 branches")

        nodes = {}
        outls = {}

        for ii in range(branch_data.shape[0]):
            branch_obj = branch_data.loc[ii]
            node_src = branch_obj.loc['node-id-src']
            node_dst = branch_obj.loc['node-id-dst']
            if node_src == node_dst:
                check_increment_dict(outls, branch_obj, node_dst)
            else:
                check_increment_dict(nodes, branch_obj, node_src)
                check_increment_dict(nodes, branch_obj, node_dst)

            if branch_data.loc[ii, 'branch-distance'] < thres and branch_data.loc[ii, 'branch-type'] == 1:

                integer_coords = tuple(skeleton_obj.path_coordinates(ii)[1:-1].T.astype(int))
                skeletony[integer_coords] = 0

                # Filter pixels with only 1 neighbor
                # integer_coords_all = tuple(skeleton_obj.path_coordinates(ii).T.astype(int))
                # degrees = skeleton_obj.degrees_image[integer_coords_all]

                # for px in range(len(integer_coords_all[0])):
                #     if degrees[px] == 1:
                #         px_tuple = (integer_coords_all[0][px], integer_coords_all[1][px])
                #         skeletony[px_tuple] = 0

                # Filter pixels in branches with 2 pixels
                # pt_idx = skeleton_obj.path(ii)
                # if len(pt_idx) == 2:
                #     for pt in range(len(pt_idx)):
                #         a = sum(x.count(pt_idx[pt]) for x in p_list)
                #         if a == 1:
                #             px_tuple = (integer_coords_all[0][pt], integer_coords_all[1][pt])
                #             skeletony[px_tuple] = 0

        # zas_img = np.zeros((w, h), dtype=np.uint8)
        # # print("Node dict", nodes)
        # single_keys = []
        # # lengs = [single_keys.append(key) if len(nodes[key]) == 3 else '' for key in nodes.keys()]
        # lengs = [single_keys.append(key) for key in outls.keys()]
        # for kk in single_keys:
        #     for nn in nodes[kk]:
        #         if nn['branch-type'] == 3:
        #             print("Branch ", nn['branch-distance'], " ", nn['branch-type'])
        #             print(nn)
        #             integer_coords = tuple(skeleton_obj.path_coordinates(nn.name)[1:-1].T.astype(int))
        #             zas_img[integer_coords] = 255
        #
        # plt.figure()
        # plt.title("Branch type 2")
        # plt.imshow(zas_img)
        # plt.show()

        skeletony = morphology.remove_small_objects(skeletony, min_size=2, connectivity=2)

        print("New skeletonize")
        skeletony = morphology.binary_dilation(skeletony)
        skeletony = morphology.skeletonize(skeletony)
        print("New skeletonize done")

        # plt.figure()
        # plt.title("Remove branches")
        # plt.imshow(skeletony)
        # plt.show()

    # print("Branch stats", bs)
    # branch_data = summarize(Skeleton(skeletony, unique_junctions=True))
    # print(branch_data.loc[0])
    # draw.overlay_euclidean_skeleton_2d(img_, branch_data)

    return skeletony




