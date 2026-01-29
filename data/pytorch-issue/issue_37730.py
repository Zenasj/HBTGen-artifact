# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn
from torchvision.models import resnet18

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = resnet18()
        self.amp_enabled = True
        self.checkpoint_enabled = True

    def forward(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        if self.checkpoint_enabled:
            x = torch.utils.checkpoint.checkpoint(base.layer1, x)
            x = torch.utils.checkpoint.checkpoint(base.layer2, x)
            x = torch.utils.checkpoint.checkpoint(base.layer3, x)
            x = torch.utils.checkpoint.checkpoint(base.layer4, x)
        else:
            x = base.layer1(x)
            x = base.layer2(x)
            x = base.layer3(x)
            x = base.layer4(x)

        x = base.avgpool(x)
        x = torch.flatten(x, 1)
        x = base.fc(x)

        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 5, 3, 128, 128
    return torch.randn(B, C, H, W, dtype=torch.float32, device='cuda')

