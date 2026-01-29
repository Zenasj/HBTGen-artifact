# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyModel, self).__init__()
        # Simulated MobileNetV2 features to match the required output shape (1280 channels)
        self.features = nn.Sequential(
            nn.Conv2d(3, 1280, kernel_size=1),  # Dummy layer to represent features
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        # Classifier with replaced Dropout (original layer 0) with Identity
        self.classifier = nn.Sequential(
            nn.Identity(),  # Replaced Dropout
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def my_model_function():
    return MyModel(num_classes=1000)

def GetInput():
    input_tensor = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    # Added as per original code's nnapi requirements
    input_tensor.nnapi_nhwc = True
    return input_tensor

