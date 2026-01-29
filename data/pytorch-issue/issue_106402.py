import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperConv(nn.Conv2d):
    def __init__(self, *args, is_lora=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_lora = is_lora

    def forward(self, *args, **kwargs):
        if self.is_lora:
            return 3 + super().forward(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)

# torch.rand(1, 3, 32, 32, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SuperConv(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = SuperConv(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def my_model_function():
    model = MyModel()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0.1)
    return model

def GetInput():
    return torch.randn(1, 3, 32, 32, dtype=torch.float32)

