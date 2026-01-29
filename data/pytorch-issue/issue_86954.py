# torch.rand(B, 3, H, W, dtype=torch.float32)  # e.g., (1, 3, 224, 224)
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Create non-contiguous input tensor (similar to the issue's problematic case)
    # Using random data with permuted dimensions to ensure non-contiguous storage
    input_data = torch.rand(224, 224, 3).permute(2, 0, 1)  # (3, 224, 224) non-contiguous
    input_data = input_data.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
    return input_data

