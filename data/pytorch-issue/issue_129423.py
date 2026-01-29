# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

def my_model_function():
    model = MyModel()
    model.eval()
    model.cuda()
    return model

def GetInput():
    return torch.randn(4, 3, 224, 224, dtype=torch.float32).to("cuda")

