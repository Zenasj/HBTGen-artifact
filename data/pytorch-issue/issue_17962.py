# torch.rand(64, 1, 28, 28, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(64, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.contiguous()  # INSERTED
        return F.log_softmax(x, dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(64, 1, 28, 28, requires_grad=True)

