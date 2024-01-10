from tensor import *


def linear(x, weight, bias):
    affine = x @ weight
    if bias is not None:
        affine = affine + bias
    return affine


class sigmoid(UnaryOperator):
    def forward(self, x):
        sigmoid = np.zeros(x.shape)
        sigmoid[x.data > 0] = 1 / (1 + np.exp(-x.data[x.data > 0]))
        sigmoid[x.data <= 0] = 1 - 1 / (1 + np.exp(x.data[x.data <= 0]))
        return sigmoid

    def grad_fn(self, x, grad):
        return self.data * (1 - self.data) * grad


class tanh(UnaryOperator):
    """Tanh运算，我们前向传播避免了溢出问题"""
    def forward(self, x):
        tanh = np.zeros(x.shape)
        tanh[x.data > 0] = 2 / (1 + np.exp(-2 * x.data[x.data > 0])) - 1
        tanh[x.data <= 0] = 1 - 2 / (1 + np.exp(2 * x.data[x.data <= 0]))
        return tanh

    def grad_fn(self, x, grad):
        return (1 - self.data**2) * grad


def relu(x):
    return maximum(0., x)


def leaky_relu(x, alpha):
    return maximum(x, alpha * x)


def softmax(x, axis=None, keepdims=False):
    """Softmax函数"""
    x_sub_max = x - x.data.max()
    exp_ = exp(x_sub_max)
    return exp_ / sum(exp_, axis=axis, keepdims=keepdims)


def log_softmax(x, axis=None, keepdims=False):
    """log-softmax函数"""
    x_sub_max = x - x.data.max()
    return x_sub_max - log(
        sum(exp(x_sub_max), axis=axis, keepdims=keepdims))


def mse_loss(y_pred, y_true, reduction='mean'):
    """均方误差"""
    square_sum = square(y_pred - y_true)
    if reduction == 'mean':
        return mean(square_sum)
    elif reduction == 'sum':
        return sum(square_sum)
    else:
        assert 0, "reduction must be mean or sum."


def nll_loss(y_pred, y_true, reduction='mean'):
    """负对数似然"""
    nll = -y_pred * y_true
    if reduction == 'mean':
        return mean(nll)
    elif reduction == 'sum':
        return sum(nll)
    else:
        assert 0, "reduction must be mean or sum."


def cross_entropy_loss(y_pred, y_true, reduction='mean'):
    """交叉熵损失"""
    update_y_pred = y_pred - y_pred.data.max()
    log_sum_exp = log(
        sum(exp(update_y_pred), 1, keepdims=True))
    nll = (log_sum_exp - update_y_pred) * y_true
    if reduction == 'mean':
        return mean(nll)
    elif reduction == 'sum':
        return sum(nll)
    else:
        assert 0, "reduction must be mean or sum."


def softmax_cross_entry(y_pred, y_true, reduction='mean'):
    pass