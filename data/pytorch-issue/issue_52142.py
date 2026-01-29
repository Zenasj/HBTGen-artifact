# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, 3, 224, 224)

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.model = MyResNet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class MyResNet18(ResNet):
    def __init__(self, num_classes):
        super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
      
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        FV = torch.flatten(x, 1)
        Logit = self.fc(FV)
        return FV, Logit

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 32
    input_shape = (batch_size, 3, 224, 224)
    return torch.randn(input_shape, dtype=torch.float32).cuda()

