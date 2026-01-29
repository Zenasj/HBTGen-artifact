# torch.rand(B, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    return model.cuda()  # Matches the issue's model setup with CUDA

def GetInput():
    return torch.rand(2, 4, dtype=torch.float32, requires_grad=True).cuda()

