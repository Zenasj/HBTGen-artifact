# torch.rand(1, 3, 32, 32, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        bn = nn.BatchNorm2d(32, affine=False)
        relu = nn.ReLU()
        dropout = nn.Dropout(0.3)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.Sequential(
            conv,
            bn,
            relu,
            dropout,
            maxpool,
        )
        self.classifier = nn.Sequential(nn.Linear(8192, 10))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

