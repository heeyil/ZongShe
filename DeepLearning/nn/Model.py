import numpy as np


class Model(object):
    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, predictions, targets):
        loss = self.loss.loss(predictions, targets)
        grad = self.loss.grad(predictions, targets)
        grads = self.net.backward(grad)
        params = self.net.get_parameters()
        steps = self.optimizer.update(grads, params)
        return loss, steps

    def update(self, steps):
        for step, (param, _) in zip(steps, self.net.get_params_and_grads()):
            for k, v in param.items():
                param[k] = param[k] + step[k]