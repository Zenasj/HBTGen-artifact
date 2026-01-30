import torch.nn as nn

from torch import nn


class MyModule(nn.Module):
    @property
    def something(self):
        hey = self.unknown_function()
        return hey


model = MyModule()
print(model.something)

class MyObject(object):
    @property
    def something(self):
        hey = self.unknown_function()
        return hey

    def __getattr__(self, name):
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))


model = MyObject()
print(model.something)