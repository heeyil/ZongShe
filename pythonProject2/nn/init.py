from numpy.random import uniform, normal
import math
import numpy as np


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndim
    assert dimensions >= 2

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1

    if dimensions > 2:
        receptive_field_size = math.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def Normal(tensor, mean=0., std=1.):
    tensor.data = normal(mean, std, tensor.shape)
    return tensor


def Uniform(tensor, low=0., high=1.0):
    tensor.data = uniform(low, high, tensor.shape)


def Constant(tensor, val):
    tensor.data = np.full(tensor.shape, fill_value=val)
    return tensor


def Zeros(tensor):
    tensor = Constant(tensor, 0.)
    return tensor


def Ones(tensor):
    tensor = Constant(tensor, 1.)
    return tensor


def XavierUniform(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6. / (fan_in + fan_out))
    tensor.data = uniform(-bound, bound, tensor.shape)
    return tensor


def XavierNormal(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    tensor.data = normal(0., std, tensor.shape)
    return tensor