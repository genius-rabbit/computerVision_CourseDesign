from data_utils import load_cifar10_data
from KNN_module import KNearestNeighbor
from cross_validation import cross_validation
import numpy as np
import os


# 加载cifar10数据
cifar_10_dir = os.path.join('dataset', 'cifar-10-batches-py')
x_train, y_train, x_test, y_test = load_cifar10_data(cifar_10_dir, use_hog=0)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# 创建用于超参数调优的交叉验证集（也可以验证集，因为数据量还是很大的）
num_training = 5000
x_tr = x_train[:num_training, ::]
x_tr = np.reshape(x_tr, (x_tr.shape[0], -1))
y_tr = y_train[:num_training]
# print(X_tr.shape, y_tr.shape)

num_testing = 1000
x_te = x_test[:num_testing, ::]
x_te = np.reshape(x_te, (x_te.shape[0], -1))
y_te = y_test[:num_testing]
# print(X_te.shape, y_te.shape)

# cross validation to get best k
k = cross_validation(x_tr, y_tr)

classify = KNearestNeighbor()
classify.train(x_tr, y_tr)

y_te_pred = classify.predict(x_te, k)
accuracy = np.sum(y_te_pred == y_te) / float(x_te.shape[0])
print('MAX K = %d, accuracy = %.3f' % (k, accuracy))

