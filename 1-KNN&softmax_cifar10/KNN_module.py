import numpy as np


class KNearestNeighbor(object):
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def compute_distances(self, x_test):
        # 计算测试集和每个训练集的欧氏距离,向量化实现需转化公式后实现（单个循环不需要）
        value_2xy = np.multiply(x_test.dot(self.x_train.T), -2)
        value_x2 = np.sum(np.square(x_test), axis=1, keepdims=True)
        value_y2 = np.sum(np.square(self.x_train), axis=1)
        dists = value_2xy + value_x2 + value_y2
        return dists

    def predict_label(self, dists, k):
        # 选择前K个距离最近的标签，从这些标签中选择个数最多的作为预测分类
        y_pred = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            # 取前K个标签
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            # 取K个标签中个数最多的标签
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

    def predict(self, x_test, k):
        dists = self.compute_distances(x_test)
        y_pred = self.predict_label(dists, k)
        return y_pred
