py
import torch
import torch.nn as nn

class BinaryStatScores(nn.Module):
    def __init__(self):
        super().__init__()


class Accuracy(nn.Module):
    def __new__(cls):
        return BinaryStatScores()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        y = Accuracy()
        return self.layer(x)


def overwrite_torch_functions():
    module_set_attr_orig = torch.nn.Module.__setattr__

    def wrap_set_attr(self, name, value):
        if isinstance(value, torch.nn.Module):
            print(value)  # <-- calls `__repr__` on the module
        module_set_attr_orig(self, name, value)

    torch.nn.Module.__setattr__ = wrap_set_attr

overwrite_torch_functions()

model = Model()
model = torch.compile(model)
model(torch.rand(2, 2))

Accuracy.__new__

BinaryStatScores

NNModuleVariable

UnspecializedNNModuleVariable

__new__

BinaryStatScores()

install_generation_tagging_init

UnspecializedNNModuleVariable

cls

nn.Module