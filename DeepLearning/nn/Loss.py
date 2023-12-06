import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    exp_sum = np.sum(exps, axis=axis, keepdims=True)
    return x - x_max -np.log(exp_sum)


class Loss(object):
    def loss(self, predicted, labels):
        raise NotImplementedError

    def grad(self, predicted, labels):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    def loss(self, predicted, labels):
        exp = np.exp(predicted - np.max(predicted, axis=1, keepdims=True))
        p = exp / np.sum(exp, axis=1, keepdims=True)
        nll = -np.log(np.sum(p * labels, axis=1))
        return np.sum(nll) / labels.shape[0]

    def grad(self, predicted, labels):
        grad = np.copy(predicted)
        grad -= labels
        return grad / labels.shape[0]


class MSE(Loss):
    def loss(self, predicted, labels):
        return 0.5 * np.sum((predicted - labels) ** 2) / labels.shape[0]

    def grad(self, predicted, labels):
        return (predicted - labels) / labels.shape[0]


class SoftmaxCrossEntropy(Loss):
    def loss(self, predicted, labels):
        nll = -(log_softmax(predicted, axis=1) * labels).sum(axis=1)
        return np.sum(nll) / labels.shape[0]

    def grad(self, predicted, labels):
        grads = softmax(predicted) - labels
        return grads / labels.shape[0]


