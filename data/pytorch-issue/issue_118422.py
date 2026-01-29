# torch.rand(2, 3, 4, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = x.to("cuda")
        prop = torch.cuda.get_device_properties(x.device)
        if prop.major == 8:
            return x + prop.multi_processor_count
        else:
            return x + prop.max_threads_per_multi_processor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

