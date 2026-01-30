import torch
import torch.nn as nn
from torch.ao.quantization import fuse_modules_qat, fuse_modules

class DummyHook:
    def _call__(self, module, module_in, module_out):
        print(f"Called from {module}")


# Declare a sub-module with well-known pattern for fusing (conv, bn, relu)
class ConvBNReLU(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=False)
        )

        
class NestedModule(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.relu1 = nn.ReLU(inplace=False)
        layers = []
        for i in range(3):
            layers.append(ConvBNReLU())
        self.features = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.features(x)


# Declare hooks in sub-sub modules
h = DummyHook()

m = NestedModule()
for sub_m in m.modules():
    for layer in sub_m.modules():
        layer.register_forward_hook(h)

# Run fuse_modules with or without quantization-aware training leads to the same error

fuse_modules_qat(m, ['features.0.0', 'features.0.1', 'features.0.2'], inplace=False)

m.eval()
fuse_modules(m, ['features.0.0', 'features.0.1', 'features.0.2'], inplace=False)