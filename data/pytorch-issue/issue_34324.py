import torch.nn as nn

from torch import nn

class MyModule(nn.Module):
    @property
    def something(self):
        hey = self.unknown_function()
        return hey


model = MyModule()
print(model.something)

class ModuleAttributeError(AttributeError):
    """ When `__getattr__` raises AttributeError inside a property,
    AttributeError is raised with the property name instead of the
    attribute that initially raised AttributeError, making the error
    message uninformative. Using `ModuleAttributeError` instead
    fixes this issue."""


ModuleAttributeError.__module__ = 'torch'