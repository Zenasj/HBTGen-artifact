# torch.rand(B, 1, H, W, dtype=torch.float32)
import torch
from torch import nn

class CounterExampleModel(nn.Module):
    def __init__(
        self,
        num_features: int = 1,
        num_classes: int = 2,
        num_hidden: int = 8,
        device: str = 'cpu',
        bias: bool = True,
    ) -> None:
        self.num_features = num_features
        self.num_classes = num_classes
        fc1 = nn.Linear(num_features, num_hidden, device=device, bias=bias)
        fc2 = nn.Linear(num_hidden, num_classes, device=device, bias=bias)
        net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            fc1,
            nn.ReLU(),
            fc2,
            nn.Softmax(dim=-1),
        )
        super().__init__()
        self.net = net
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        return self.net(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CounterExampleModel()
        self.fc1 = self.model.fc1
        self.fc2 = self.model.fc2

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 2, 1, 3, 4  # Example dimensions matching input shape requirements
    return torch.rand(B, C, H, W, dtype=torch.float32)

