from .module import Module
from ..parameter import Parameter
from .. import functional as F
from tensor import empty
from ..init import XavierUniform, Zeros


class Linear(Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(empty((self.in_features, self.out_features), dtype))
        self.bias = Parameter(empty(self.out_features, dtype)) if bias else None

        self.init_parameters()

    def init_parameters(self):
        XavierUniform(self.weight)
        if self.bias is not None:
            Zeros(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)