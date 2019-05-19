import os
import random
import numpy as np
import matplotlib.image as mpimg

'''
data_set : 
    car,bird,turtle,dog,lizard
    size: 128 * 128
    image_num = 180
    box_format: index, xmin,ymin,xmax,ymax
'''

# load data
root_data_dir = os.path.join('data_set', 'tiny_vid')

data_bird_img_dir = os.path.join(root_data_dir, 'bird')
data_bird_box_file = os.path.join(root_data_dir, 'bird_gt.txt')

data_car_img_dir = os.path.join(root_data_dir, 'car')
data_car_box_file = os.path.join(root_data_dir, 'car_gt.txt')

data_dog_img_dir = os.path.join(root_data_dir, 'dog')
data_dog_box_file = os.path.join(root_data_dir, 'dog_gt.txt')

data_lizard_img_dir = os.path.join(root_data_dir, 'lizard')
data_lizard_box_file = os.path.join(root_data_dir, 'lizard_gt.txt')

data_turtle_img_dir = os.path.join(root_data_dir, 'turtle')
data_turtle_box_file = os.path.join(root_data_dir, 'turtle_gt.txt')


def data_load():
    train_img_set_x = []
    train_img_set_y = []
    train_box_set = []
    test_img_set_x = []
    test_img_set_y = []
    test_box_set = []

    file_obj = [open(data_bird_box_file, 'r'),
                open(data_car_box_file, 'r'),
                open(data_dog_box_file, 'r'),
                open(data_lizard_box_file, 'r'),
                open(data_turtle_box_file, 'r')]
    img_dir = [data_bird_img_dir,
               data_car_img_dir,
               data_dog_img_dir,
               data_lizard_img_dir,
               data_turtle_img_dir]

    for obj in file_obj:
        for index in range(180):
            line = [int(x) for x in obj.readline().strip().split()]
            if index < 150:
                train_box_set.append(line[1:])
            else:
                test_box_set.append(line[1:])

    type_flag = 0
    for dir_path in img_dir:
        img_list = os.listdir(dir_path)
        index = 0

        for img in img_list:
            if index < 150:
                train_img_set_x.append(mpimg.imread(os.path.join(dir_path, img)))
                train_img_set_y.append(type_flag)
            elif index >= 150 and not index >= 180:
                test_img_set_x.append(mpimg.imread(os.path.join(dir_path, img)))
                test_img_set_y.append(type_flag)
            index += 1
        type_flag += 1

    x_img_tr = np.array(train_img_set_x).transpose((0, 3, 1, 2))
    y_img_tr = np.array(train_img_set_y)
    box_tr = np.array(train_box_set)
    # print(box_tr)
    box_tr = box_tr / 128.
    x_img_te = np.array(test_img_set_x).transpose((0, 3, 1, 2))
    y_img_te = np.array(test_img_set_y)
    box_te = np.array(test_box_set)
    box_te = box_te / 128.

    # random dataset
    randomIndices = random.sample(range(x_img_tr.shape[0]), x_img_tr.shape[0])
    x_img_tr = x_img_tr[randomIndices]
    y_img_tr = y_img_tr[randomIndices]
    box_tr = box_tr[randomIndices]

    randomIndices = random.sample(range(x_img_te.shape[0]), x_img_te.shape[0])
    x_img_te = x_img_te[randomIndices]
    y_img_te = y_img_te[randomIndices]
    box_te = box_te[randomIndices]

    print('data set shape:')
    print('train x: ', x_img_tr.shape, '\ttrain y: ', y_img_tr.shape, '\ttrain box: ', box_tr.shape)
    print('test x: ', x_img_te.shape, '\ttest y: ', y_img_te.shape, '\ttest box: ', box_te.shape)
    return x_img_tr, y_img_tr, box_tr, x_img_te, y_img_te, box_te


def calc_ious(ex_rois, gt_rois):
    ex_area = (1. + ex_rois[:, 2] - ex_rois[:, 0]) * (1. + ex_rois[:, 3] - ex_rois[:, 1])
    gt_area = (1. + gt_rois[:, 2] - gt_rois[:, 0]) * (1. + gt_rois[:, 3] - gt_rois[:, 1])
    area_sum = ex_area + gt_area

    lb = np.maximum(ex_rois[:, 0], gt_rois[:, 0])
    rb = np.minimum(ex_rois[:, 2], gt_rois[:, 2])
    tb = np.maximum(ex_rois[:, 1], gt_rois[:, 1])
    ub = np.minimum(ex_rois[:, 3], gt_rois[:, 3])

    width = np.maximum(1. + rb - lb, 0.)
    height = np.maximum(1. + ub - tb, 0.)
    area_i = width * height
    area_u = area_sum - area_i

    ious = area_i / area_u
    return ious.sum() / ex_rois.shape[0]
