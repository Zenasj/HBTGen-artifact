# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Assuming standard image input (e.g., CIFAR-10)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(16 * 8 * 8, 10)  # 32x32 input → 8x8 after pooling

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor with requires_grad=True to match the training loop's usage
    return torch.rand(4, 3, 32, 32, dtype=torch.float32, requires_grad=True)

