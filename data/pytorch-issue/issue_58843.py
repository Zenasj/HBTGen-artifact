# torch.rand(3, 2, 1, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(2, affine=True, track_running_stats=True)
        self.batch_norm = nn.BatchNorm2d(2, affine=True, track_running_stats=True)
    
    def forward(self, x):
        instance_out = self.instance_norm(x)
        batch_out = self.batch_norm(x)
        return instance_out == batch_out  # Element-wise comparison tensor

def my_model_function():
    model = MyModel()
    model.eval()  # Matches evaluation mode setup in the original issue's test
    return model

def GetInput():
    return torch.randn(3, 2, 1, 3)  # Matches input shape from the issue's reproduction code

