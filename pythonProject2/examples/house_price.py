import pandas as pd
from utils.data.dataloader import dataloader
from nn.modules import linear, loss
from optim.adam import Adam
from nn.modules.module import Module
from utils.train_metric import *

# 读取数据
train_data = pd.read_csv(r"D:/dataset/train.csv")
test_data = pd.read_csv(r"D:/dataset/test.csv")
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化数据并用0填充缺失项
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散型特征-独热编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# 将dataframe数据转换为numpy型数据
n_train = train_data.shape[0]
train_features = Tensor(all_features[:n_train].values, dtype=float)
test_features = Tensor(all_features[n_train:].values, dtype=float)
train_labels = Tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=float)


class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = linear.Linear(train_features.shape[1], 1)

    def forward(self, x):
        fc = self.fc1(x)
        return fc


net = Net()
loss = loss.MSELoss()
optimizer = Adam(net.parameters(), lr=5)

model = train_test(net, loss, optimizer)
model.train(train_features, train_labels,  64, 50, True, True)
predictions = model.predict(test_features)
