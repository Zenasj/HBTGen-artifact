# torch.rand(3, dtype=torch.float32)
import torch
import torch.nn as nn

class IndexPutModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.indices = torch.tensor([0, 0], device=device)
        self.values = torch.tensor(1.0, device=device)

    def forward(self, x):
        x = x.to(self.device)
        return x.index_put_([self.indices], self.values, accumulate=True)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpu_model = IndexPutModule('cpu')
        self.gpu_model = IndexPutModule('cuda')

    def forward(self, x):
        cpu_out = self.cpu_model(x)
        try:
            gpu_out = self.gpu_model(x)
            return torch.allclose(cpu_out, gpu_out)
        except:
            return torch.tensor(False, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros(3, dtype=torch.float32)

