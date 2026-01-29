# torch.rand(1, 1, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm2d(1, momentum=0.5)  # Problematic layer with specified momentum
        
    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Returns the model instance in eval mode (as per original issue's setup)
    model = MyModel()
    model.eval()
    return model

def GetInput():
    # Generate input matching the model's expected shape
    return torch.rand(1, 1, 224, 224, dtype=torch.float32)

