import numpy as np


# softmax运算
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)