from nn.modules.module import Module
from collections import OrderedDict


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __iter__(self):
        return iter(self._modules.values())

    def __add__(self, other):
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret

    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def __len__(self):
        return len(self.modules)


class ModuleList(Module):
    pass


class ModuleDict(Module):
    pass