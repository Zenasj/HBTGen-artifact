# torch.rand(512, 64, 768, dtype=torch.float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(768, 768, bias=True)
    
    def forward(self, x):
        return self.fc1(x)

def my_model_function():
    model = MyModel()
    model.cuda()
    model.half()
    return model

def GetInput():
    return torch.rand(512, 64, 768, dtype=torch.float16, device='cuda', requires_grad=True)

