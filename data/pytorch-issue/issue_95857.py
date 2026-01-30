import torch.nn as nn

import weakref

import torch


class Trainer:
    def __init__(self):
        self.foo = True


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainer = None

    def forward(self):
        ...

    def log(self):
        if self.trainer.foo is None:
            raise RuntimeError
        print("Good!")


model = MyModel()
trainer = Trainer()
model.trainer = weakref.proxy(trainer)

# works
model.log()

compiled_log = torch.compile(model.log)
# fails
compiled_log()