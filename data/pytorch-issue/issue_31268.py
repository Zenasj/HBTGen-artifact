# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torchvision import models

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.submodule = models.resnet50()  # Submodule causing the issue when cloned after JIT serialization

    def forward(self, x):
        return self.submodule(x)

def my_model_function():
    # Return the model in eval mode as in the original issue's reproduction steps
    model = MyModel()
    model.eval()
    return model

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

