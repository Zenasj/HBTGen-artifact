# torch.rand(2, 3, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

def copy_as_numpy(tensor):
    return tensor.cpu().numpy()  # Added as per issue's proposed shorthand
torch.Tensor.np = copy_as_numpy  # Inject method into Tensor class

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)  # 14x14 after pooling

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 28, 28, dtype=torch.float32)

