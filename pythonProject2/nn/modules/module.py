from collections import OrderedDict
from nn.parameter import Parameter


class Module:
    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def __call__(self, *x):
        return self.forward(*x)

    def __setattr__(self, __name, __value):
        self.__dict__[__name] = __value
        if isinstance(__value, Parameter):
            self._parameters[__name] = __value
        if isinstance(__value, Module):
            self._modules[__name] = __value
            for key in __value._parameters:
                self._parameters[__name + "." + key] = __value._parameters[key]

    def parameters(self):
        for param in self._parameters.values():
            yield param

    def add_module(self, name, module):
        self._modules[name] = module

    def modules(self):
        for module in self._modules.values():
            yield module

    def train(self, mode=True):
        self.training = mode
        for module in self.modules():
            module.train(mode)

    def forward(self, x):
        raise NotImplementedError

    def eval(self):
        return self.train(False)