import numpy as np
import nn.Model as Model
from utils.data.dataloader import dataloader
from utils.visualization import Animator
from utils import evaluate


class train_test(object):

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.model = Model.Model(self.net, self.loss, self.optimizer)

    # 定义训练方法
    def train(self, train_features, train_labels, batch_size, epochs, shuffle=True, log_rmse=False):

        data_iter = dataloader(batch_size, shuffle)

        animator = Animator(xlabel='epoch', ylabel='loss', legend=['train_loss'], xlim=[1, epochs])

        for epoch in range(epochs):
            for X, y in data_iter(train_features, train_labels):
                pre = self.model.forward(X)
                loss, steps = self.model.backward(pre, y)
                self.model.update(steps)

            predictions = self.model.forward(train_features)

            if log_rmse:
                clipped_preds = np.clip(predictions, 1, float('inf'))
                clipped_preds = clipped_preds.astype('float')
                train_loss, _ = self.model.backward(np.log(clipped_preds), np.log(train_labels))

            else:
                train_loss, _ = self.model.backward(predictions, train_labels)

            animator.add(epoch + 1, train_loss)
            print(f'epoch {epoch + 1}, train_loss {train_loss : f}')

    # 定义测试方法
    def test(self, batch_size, test_features, test_labels, shuffle=True):

        data_iter = dataloader(batch_size, shuffle)
        batch_num = len(test_features) // batch_size + 1
        i = 0
        animator = Animator(xlabel='batch', ylabel='accuracy', xlim=[1, batch_num],
                            ylim=[0, 0.5], legend=['test_accuracy'])
        metric = evaluate.Accumulator(2)
        for X, y in data_iter(test_features, test_labels):
            pre = self.model.forward(X)
            metric.add(evaluate.accuracy(pre, y), len(y))
            test_acc = metric[0] / metric[1]

            animator.add(i + 1, test_acc)
            i += 1
            print(f'test_Acc {test_acc:f}')

    # 定义预测函数
    def predict(self, features):
        predictions = self.model.forward(features)
        return predictions
