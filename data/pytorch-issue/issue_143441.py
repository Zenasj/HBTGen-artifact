import torch.nn as nn

from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 4)


model = SimpleModel()
model.set_submodule('0', nn.Conv2d(2, 3, 3))
# this will work, despite there being no module named '0' in model
assert isinstance(getattr(model, '0'), nn.Conv2d)


try:
    model.set_submodule('foo.bar', nn.Conv2d(2, 3, 3))
except AttributeError as e:
    message = str(e)
    assert message == 'SimpleModel has no attribute `foo`'