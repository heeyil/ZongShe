import numpy as np
import pandas as pd
from nn import Loss
from nn.Sequential import Sequential
from nn import Optimizer
from nn import Layer
from utils import math
from utils.data import datapre
from utils.train_test import train_test


# 导入图片数据
train_data = pd.read_csv(r"D:/dataset/archive/fashion-mnist_train.csv")
test_data = pd.read_csv(r"D:/dataset/archive/fashion-mnist_test.csv")

# 处理训练数据集
train_labels = datapre.one_hot(train_data.iloc[:, 0].values.reshape(-1, 1), 10)
train_features = train_data.iloc[:, 1:].values.reshape(-1, 784).astype(np.float32) / 255


# 处理测试集
test_labels = datapre.one_hot(test_data.iloc[:, 0].values.reshape(-1, 1), 10)
test_features = test_data.iloc[:, 1:].values.reshape(-1, 784).astype(np.float32) / 255


# 生成模型
loss = Loss.SoftmaxCrossEntropy()
optimizer = Optimizer.Momentum(0.1)
net = Sequential([Layer.Linear(784, 128),
                  Layer.ReLu(),
                  Layer.Linear(128, 10)
                  ])

# 训练模型
model = train_test(net, loss, optimizer)
model.train(train_features, train_labels, 256, 10, True, False)

# 用训练集测试测试模型
model.test(256, test_features, test_labels, True)
