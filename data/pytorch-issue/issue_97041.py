# torch.rand(B, 3, 224, 224, dtype=torch.float32, device='cuda')  # Input shape for ResNet-18
import torch
from torch import nn
from torchvision.models import resnet18

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model1 = resnet18().cuda()  # torchvision's ResNet-18 (failing model)
        self.model2 = nn.Sequential(
            nn.Linear(3*224*224, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1)
        ).cuda()  # Hand-crafted model (working model)

    def forward(self, x):
        # Compare execution success between the two models
        success1 = False
        success2 = False
        
        try:
            _ = self.model1(x)
            success1 = True
        except Exception:
            pass
        
        try:
            _ = self.model2(x.view(x.size(0), -1))
            success2 = True
        except Exception:
            pass
        
        # Return True if one succeeded and the other failed (indicates difference)
        return torch.tensor(success1 != success2, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input matching ResNet-18's expected shape on CUDA
    return torch.rand(1, 3, 224, 224, dtype=torch.float32, device='cuda')

