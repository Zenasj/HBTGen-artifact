# torch.rand(16, 3, 128, 128, dtype=torch.float32)
import torch
import torchvision.models as models

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121().to(torch.float32).cuda()
    
    def forward(self, x):
        return self.densenet(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(16, 3, 128, 128, dtype=torch.float32).cuda()

