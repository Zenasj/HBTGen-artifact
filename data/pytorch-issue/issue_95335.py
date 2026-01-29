# torch.rand(512, 64, 768, dtype=torch.float16)  # Input shape and dtype
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(config['input_units'], config['hidden_units'], bias=True)
        
    def forward(self, x):
        out = self.fc1(x)
        return out

def my_model_function():
    config = {'input_units': 768, 'hidden_units': 768}
    model = MyModel(config)
    model = model.half().cuda()  # Matches original repro's .half() and .cuda()
    return model

def GetInput():
    input_shape = [512, 64, 768]  # From config in the issue
    return torch.rand(input_shape, dtype=torch.float16, device='cuda', requires_grad=True)

