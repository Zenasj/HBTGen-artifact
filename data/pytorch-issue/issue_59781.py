# torch.rand(1, 3, 32, 32, dtype=torch.float32)  # Inferred input shape for CIFAR-10

import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.qat_resnet18 = models.resnet18(pretrained=True)
        self.qat_resnet18.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_fake_quant,
            weight=torch.quantization.default_per_channel_weight_fake_quant,
        )
        torch.quantization.prepare_qat(self.qat_resnet18, inplace=True)
        self.qat_resnet18.apply(torch.quantization.enable_observer)
        self.qat_resnet18.apply(torch.quantization.enable_fake_quant)

    def forward(self, x):
        return self.qat_resnet18(x)

def my_model_function():
    model = MyModel()
    dummy_input = torch.randn(16, 3, 224, 224)
    _ = model(dummy_input)
    for module in model.modules():
        if isinstance(module, torch.quantization.FakeQuantize):
            module.calculate_qparams()
    model.apply(torch.quantization.disable_observer)
    return model

def GetInput():
    return torch.randn(1, 3, 32, 32)  # Adjusted to match the input expected by MyModel

