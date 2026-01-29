# torch.randint(-8192, 128, (3,), dtype=torch.int64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, repeats=2):  # Reduced repeats for testability; original issue used 433894953 causing OOM
        super(MyModel, self).__init__()
        self.repeats = repeats

    def forward(self, x):
        return x.repeat_interleave(self.repeats)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(-8192, 128, (3,), dtype=torch.int64)

