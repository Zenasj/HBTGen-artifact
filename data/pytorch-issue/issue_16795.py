# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, 3, 224, 224)
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        resnet50 = models.resnet18(pretrained=True)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 104),  # Assuming 104 classes for age prediction
            nn.LogSoftmax(dim=1)
        )
        self.model = resnet50

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 32
    channels = 3
    height = 224
    width = 224
    input_tensor = torch.rand(batch_size, channels, height, width, dtype=torch.float32)
    return input_tensor

