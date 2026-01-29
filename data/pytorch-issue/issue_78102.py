# torch.rand(1, 2, 8, 8), torch.rand(1, 3, 6, 6), torch.rand(1, 2, 5)
import torch
import torch.nn as nn
import torch.nn.functional as F

class QAvgPool2dModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant1 = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def forward(self, x):
        res = F.avg_pool2d(self.quant1(x), kernel_size=2, stride=1, padding=0)
        return self.dequant(res)

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = torch.ao.quantization.default_qconfig
        self.fc1 = torch.ao.quantization.QuantWrapper(
            torch.nn.Conv2d(3, 5, 2, bias=True).to(dtype=torch.float)
        )
    
    def forward(self, x):
        return self.fc1(x)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = torch.ao.quantization.default_qconfig
        self.fc1 = torch.ao.quantization.QuantWrapper(
            torch.nn.Linear(5, 10).to(dtype=torch.float)
        )
    
    def forward(self, x):
        return self.fc1(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = QAvgPool2dModule()
        self.conv = ConvModel()
        self.linear = LinearModel()
    
    def forward(self, inputs):
        # Run all submodels and return outputs as tuple (assumed comparison during export)
        x_avg, x_conv, x_linear = inputs
        return (
            self.pool(x_avg),
            self.conv(x_conv),
            self.linear(x_linear)
        )

def my_model_function():
    # Initialize all submodels with quantization configuration
    model = MyModel()
    for submodule in [model.pool, model.conv, model.linear]:
        submodule.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    return model

def GetInput():
    # Generate inputs matching all submodel requirements
    return (
        torch.rand(1, 2, 8, 8),
        torch.rand(1, 3, 6, 6),
        torch.rand(1, 2, 5)
    )

