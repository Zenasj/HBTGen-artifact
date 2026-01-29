import torch
import torch.nn as nn
import torchvision.models as models

# torch.rand(B, 3, 256, 256, dtype=torch.float32)  # 3 channels, resized to 256x256
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        # Adjust final layer for 2 classes (assumed from issue's dataset setup)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.base_model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 4 (from DataLoader), 3 channels, 256x256 (resize transform)
    return torch.rand(4, 3, 256, 256, dtype=torch.float32)

