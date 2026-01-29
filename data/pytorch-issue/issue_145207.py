# torch.rand(24, 1, 4, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.m3 = nn.Linear(in_features=1, out_features=43, bias=True)
        self.m4 = nn.MaxPool2d(kernel_size=(2, 42), stride=2, padding=0, dilation=1, ceil_mode=False)
        self.m5 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.m3(x)
        x = self.m4(x)
        x = self.m5(x)
        return x

def my_model_function():
    model = MyModel()
    model.eval()  # Ensure BatchNorm uses running stats (critical for comparison)
    return model

def GetInput():
    return torch.rand(24, 1, 4, 1, dtype=torch.float32)

