import torch
import torch.nn as nn

# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred input shape from MobileNetV2 use case
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified MobileNetV2-like architecture based on common Android use cases
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.ReLU6(),
            nn.Conv2d(16, 24, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, groups=24),
            nn.ReLU6(),
            nn.Conv2d(24, 32, kernel_size=1),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1)  # Simplified global average pooling
        )
        self.classifier = nn.Linear(32, 1000)  # Example output size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def my_model_function():
    # Returns quantization-aware instance (as hinted by "mobNet2Quant" in issue)
    model = MyModel()
    # Placeholder for quantization setup (since original issue references quantized models)
    # model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # torch.quantization.prepare_qat(model, inplace=True)
    # torch.quantization.convert(model, inplace=True)
    return model

def GetInput():
    # Returns random input matching MobileNetV2's expected dimensions
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

