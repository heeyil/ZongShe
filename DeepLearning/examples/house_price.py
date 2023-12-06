import pandas as pd
from nn import Loss
from nn import Sequential
from nn import Optimizer
from nn import Layer
from utils.data import datapre
from utils.train_test import train_test


# 读取数据
train_data = pd.read_csv(r"D:/dataset/train.csv")
test_data = pd.read_csv(r"D:/dataset/test.csv")
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化数据并用0填充缺失项
data = datapre.feature_scaling(all_features)
data.normalization()
data.fillnan(0)
all_features = data.re()

# 处理离散型特征-独热编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# 将dataframe数据转换为numpy型数据
n_train = train_data.shape[0]
train_features = all_features[:n_train].values
test_features = all_features[n_train:].values
train_labels = train_data.SalePrice.values.reshape(-1, 1)

# -----------------------------------------------------------------------------------------------------

# 定义模型
loss = Loss.MSE()
# 在优化器的实例化时输入学习率
optimizer = Optimizer.RMSProp(5)
net = Sequential.Sequential([Layer.Linear(train_features.shape[1], 1)])

# 训练模型
model = train_test(net, loss, optimizer)
model.train(train_features, train_labels,  64, 50, True, True)
# 应用测试集输出预测结果
predictions = model.predict(test_features)
