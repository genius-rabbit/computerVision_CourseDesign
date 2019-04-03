from data_utils import load_cifar10_data
import matplotlib.pyplot as plt
from soft_max_module import SoftMax
import numpy as np
import os


# 加载cifar10数据
cifar_10_dir = os.path.join('dataset', 'cifar-10-batches-py')
x_train, y_train, x_test, y_test = load_cifar10_data(cifar_10_dir, features='hog')
print('start to train')

num_training = 50000
x_tr = x_train[:num_training, ::]
x_tr = np.reshape(x_tr, (x_tr.shape[0], -1))
y_tr = y_train[:num_training]

num_testing = 10000
x_te = x_test[:num_testing, ::]
x_te = np.reshape(x_te, (x_te.shape[0], -1))
y_te = y_test[:num_testing]

classify = SoftMax(batch_size=32, learn_rate=0.1, epoch=150)
test_acces, test_losses, train_losses = classify.train(x_tr, y_tr, x_te, y_te)

plt.errorbar(range(len(test_acces)), test_acces)
plt.title('softmax')
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.show()
exit()
