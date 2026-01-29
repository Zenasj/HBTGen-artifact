# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn
from torch.distributed._tensor.device_mesh import DeviceMesh
import os

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Reproduce environment setup from the issue example
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '25364'
        # Create problematic DeviceMesh instance (triggers NCCL error on CPU)
        self.device_mesh = DeviceMesh("cpu", torch.arange(1))
    
    def forward(self, x):
        # Dummy forward to satisfy model interface requirements
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

