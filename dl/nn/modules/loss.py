from .module import Module
from .. import functional as F
from tensor import Tensor


class Loss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        assert self.reduction in {'mean', 'sum'}

    def forward(self, y_pred, y_true):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true, reduction=self.reduction)


class NLLLoss(Loss):
    def forward(self, y_pred, y_true):
        return F.nll_loss(y_pred, y_true, reduction=self.reduction)


class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        return F.cross_entropy_loss(y_pred, y_true, reduction=self.reduction)


class SoftmaxCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        pass