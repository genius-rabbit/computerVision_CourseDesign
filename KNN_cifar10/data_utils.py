import pickle
import numpy as np
import os
from skimage import feature as ft
from skimage import color


def hog_extraction(data):
    num = data.shape[0]
    # 提取训练样本的HOG特征
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


def load_cifar10_data(root, use_hog):
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

    if use_hog:
        print('use hog to get img feature')
        Xtr = hog_extraction(Xtr)
        Xte = hog_extraction(Xte)
    return Xtr, Ytr, Xte, Yte
