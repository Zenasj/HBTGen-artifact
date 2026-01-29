# torch.rand(256, 100000, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Run multinomial on both CPU and CUDA, compare outputs
        cpu_out = torch.multinomial(x.to('cpu'), 1)  # CPU path
        cuda_out = torch.multinomial(x.to('cuda'), 1)  # CUDA path
        # Compare results and return difference indicator (1 if differ, 0 otherwise)
        return (cpu_out != cuda_out.to('cpu')).any().float()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(256, 100000, dtype=torch.float32)

