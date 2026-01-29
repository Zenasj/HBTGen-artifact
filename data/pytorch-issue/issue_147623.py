# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (batch_size, 3, 32, 32)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, params):
        super(MyModel, self).__init__()
        self.num_classes = params['num_classes']
        self.confidence_threshold = params['confidence_threshold']
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.early_exit1 = nn.Linear(16384, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.maxpool1(x)
        x = self.relu(x)

        return x, nn.functional.softmax(self.early_exit1(x.clone().view(x.size(0), -1)), dim=1)

def my_model_function():
    params = {
        'num_classes': 10,
        'confidence_threshold': 0.5
    }
    return MyModel(params)

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    batch_size = 32
    input_shape = (batch_size, 3, 32, 32)
    return torch.rand(input_shape, dtype=torch.float32).to('cuda')

