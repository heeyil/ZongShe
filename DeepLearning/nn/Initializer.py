import numpy as np


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


class Initializer(object):

    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    def init(self, shape):
        raise NotImplementedError


class Normal(Initializer):

    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def init(self, shape):
        return np.random.normal(loc=self._mean, scale=self._std, size=shape)


class Uniform(Initializer):

    def __init__(self, a=0.0, b=1.0):
        self._a = a
        self._b = b

    def init(self, shape):
        return np.random.uniform(low=self._a, high=self._b, size=shape)


class Constant(Initializer):

    def __init__(self, val):
        self._val = val

    def init(self, shape):
        return np.full(shape=shape, fill_value=self._val)


class Zeros(Constant):

    def __init__(self):
        super(Zeros, self).__init__(0.0)


class Ones(Constant):

    def __init__(self):
        super(Ones, self).__init__(1.0)


class XavierUniform(Initializer):

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low=-a, high=a, size=shape)


class XavierNormal(Initializer):

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = get_fans(shape)
        std = self._gain * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(loc=0.0, scale=std, size=shape)
