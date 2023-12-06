import numpy as np
from nn import Sequential
from nn import Layer
from nn import Model
from nn import Loss
from nn import Optimizer
from utils.data.dataloader import dataloader
from utils.visualization import Animator

num_examples = 1000
true_w = np.array([2, -3.4])
true_b = 4.2
features = np.random.normal(0, 1, size=(num_examples, len(true_w)))
labels = np.dot(true_w, features.T) + true_b
labels += np.random.normal(0, 0.01, labels.shape)
labels = labels.reshape(1000, 1)

data_iter = dataloader(10, True)
net = Sequential.Sequential([Layer.Linear(2, 1)])
loss = Loss.MSE()
optimizer = Optimizer.Adam(lr=0.01)


num_epochs = 10
Loss = []

model = Model.Model(net, loss, optimizer)

animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], legend=['train_loss'])

for epoch in range(num_epochs):
    for X, y in data_iter(features, labels):
        predictions = model.forward(X)
        loss, steps = model.backward(predictions, y)
        model.update(steps)

    pre = model.forward(features)
    l, _ = model.backward(pre, labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    animator.add(epoch + 1, l)