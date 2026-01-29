# torch.rand(2, 3, 224, 224, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torchvision

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet18 = torchvision.models.quantization.resnet.resnet18()
        self.mobilenet_v2 = torchvision.models.quantization.mobilenet_v2()
        self.fuse_models()
        self.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        self.prepare_qat()

    def fuse_models(self):
        self.resnet18.fuse_model()
        self.mobilenet_v2.fuse_model()

    def prepare_qat(self):
        torch.quantization.prepare_qat(self.resnet18, inplace=True)
        torch.quantization.prepare_qat(self.mobilenet_v2, inplace=True)

    def forward(self, x):
        y_resnet18 = self.resnet18(x)
        y_mobilenet_v2 = self.mobilenet_v2(x)
        return y_resnet18, y_mobilenet_v2

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

