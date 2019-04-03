import pickle
import numpy as np
import os
from skimage import feature as ft
from skimage import color


# histogram of oriented gradient, 梯度方向直方图特征
def hog_extraction(data):
    num = data.shape[0]
    data1_hogfeature = []

    for i in range(num):
        img = data[i].reshape(32, 32, 3)
        gray = color.rgb2gray(img)
        # 转化为array
        gray_array = np.array(gray)

        # 提取HOG特征
        hogfeature = ft.hog(gray_array, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False, transform_sqrt=True)
        data1_hogfeature.append(hogfeature)

    # 把data1_hogfeature中的特征按行堆叠
    data_hogfeature = np.reshape(np.concatenate(data1_hogfeature), [num, -1])
    return data_hogfeature


# LBP(local binary pattern)是一种用来描述图像局部纹理特征的算子
def lbp_extraction(images_data, hist_size=256, lbp_radius=1, lbp_point=8):
    n_images = images_data.shape[0]
    hist = np.zeros((n_images, hist_size))

    for i in np.arange(n_images):
        img = images_data[i].reshape(32, 32, 3)
        gray = color.rgb2gray(img)
        gray_array = np.array(gray)

        lbp = ft.local_binary_pattern(gray_array, lbp_point, lbp_radius, 'default')
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist size:256
        hist[i], _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins))

    return hist


def load_cifar_batch(filename):
    """ load single batch of cifar """
    print(filename)
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_cifar10_data(root, features):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        x, y = load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_cifar_batch(os.path.join(root, 'test_batch'))

    if features == 'hog':
        print('use hog to get img feature')
        Xtr = hog_extraction(Xtr)
        Xte = hog_extraction(Xte)
    elif features == 'lbp':
        print('use lbp to get img feature')
        Xtr = lbp_extraction(Xtr)
        Xte = lbp_extraction(Xte)
    return Xtr, Ytr, Xte, Yte

