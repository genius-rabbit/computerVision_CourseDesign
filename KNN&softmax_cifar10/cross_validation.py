from KNN_module import KNearestNeighbor
import matplotlib.pyplot as plt
import numpy as np


def cross_validation(x_train, y_train):
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_accuracy = {}
    # 将数据集分为5份
    x_train_folds = np.array_split(x_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    # 计算每种K值
    for k in k_choices:
        k_accuracy[k] = []
        # 每个K值分别计算每份数据集作为测试集时的正确率
        for index in range(num_folds):
            # 构建数据集
            x_te = x_train_folds[index]
            y_te = y_train_folds[index]
            x_tr = np.vstack(x_train_folds[:index] + x_train_folds[index + 1:])
            y_tr = np.hstack(y_train_folds[:index] + y_train_folds[index + 1:])
            # 预测结果
            classify = KNearestNeighbor()
            classify.train(x_tr, y_tr)
            y_te_pred = classify.predict(x_te, k=k)
            accuracy = np.sum(y_te_pred == y_te) / float(x_te.shape[0])
            k_accuracy[k].append(accuracy)

    for k, accuracy_list in k_accuracy.items():
        for accuracy in accuracy_list:
            print("k = %d, accuracy = %.3f" % (k, accuracy))

    # 可视化K值效果
    for k in k_choices:
        accuracies = k_accuracy[k]
        plt.scatter([k] * len(accuracies), accuracies)
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_accuracy.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_accuracy.items())])
    # 根据均值和方差构建误差棒图
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    index = np.argmax(accuracies_mean).__int__()
    return k_choices[index]
