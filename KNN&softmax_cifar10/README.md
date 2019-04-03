# KNN & softmax

## file lists

- data_utils.py：加载数据集，对数据集进行特征提取等等
- cross-validation.py：将训练集分成五份，进行交叉验证得到最优的k值，并返回
- KNN_module.py：KNN模型，包括训练，预测
- KNN：**KNN训练的主函数**，进行数据集加载，交叉验证得到最优的K，然后得到测试机的准确率
- soft_max_module.py：softmax模型，包括计算loss，dW的计算，准确率的计算，W的更新
- soft_max：**softmax模型的主函数**，进行数据集加载，W的训练，测试集正确率计算

