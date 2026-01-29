# torch.randint(10, (6,), dtype=torch.int64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        y = [x[0], x[2], x[4]]  # Extract elements at indices 0, 2, 4
        return torch.LongTensor(y)  # Fails with Dynamo due to fake tensor issues

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(10, (6,), dtype=torch.int64)

