import pandas as pd
from tensor import *
from utils.data.dataloader import dataloader
from nn.modules import linear, activation, loss
from nn.modules.container import Sequential
from optim.adam import Adam
from utils.visual import Animator
from nn.modules.module import Module

# 导入图片数据
train_data = pd.read_csv(r"D:/dataset/archive/fashion-mnist_train.csv")
test_data = pd.read_csv(r"D:/dataset/archive/fashion-mnist_test.csv")


def one_hot(targets, n_classes):
    return np.eye(n_classes, dtype=np.float32)[np.array(targets).reshape(-1)]


# 处理训练数据集
train_labels = Tensor(one_hot(train_data.iloc[:, 0].values.reshape(-1, 1), 10), dtype=float)
train_features = Tensor(train_data.iloc[:, 1:].values.reshape(-1, 784).astype(np.float32) / 255, dtype=float)

# 处理测试集
test_labels = one_hot(test_data.iloc[:, 0].values.reshape(-1, 1), 10)
test_features = test_data.iloc[:, 1:].values.reshape(-1, 784).astype(np.float32) / 255

