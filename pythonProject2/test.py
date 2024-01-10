from tensor import *
from utils.data.dataloader import dataloader
from nn.modules import linear, activation, loss
from nn.modules.container import Sequential
from optim.adam import Adam
from utils.visual import Animator
from nn.modules.module import Module

num_examples = 1000
true_w = np.array([2, -3.4])
true_b = 4.2
features = np.random.normal(0, 1, size=(num_examples, len(true_w)))
labels = np.dot(true_w, features.T) + true_b
labels += np.random.normal(0, 0.01, labels.shape)
labels = labels.reshape(1000, 1)

data_iter = dataloader(10, True)

nett = Sequential(linear.Linear(2, 1))


class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = linear.Linear(2, 1)

    def forward(self, x):
        fc = self.fc1(x)
        return fc


net = Net()
loss = loss.MSELoss()
optimizer = Adam(net.parameters(), lr=0.01)

num_epochs = 10

animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], legend=['train_loss'])

for epoch in range(num_epochs):
    net.train()
    for X, y in data_iter(features, labels):
        output = nett.forward(Tensor(X))
        l = loss(output, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    net.eval()
    output = net.forward(Tensor(features))
    L = loss(output, labels).data
    print(f'epoch {epoch + 1}, loss {L:f}')
    animator.add(epoch+1, L.item())