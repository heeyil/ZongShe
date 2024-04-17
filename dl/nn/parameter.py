from tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(
            data=data.data,
            dtype=data.dtype,
            requires_grad=True
        )

    def to(self):
        pass