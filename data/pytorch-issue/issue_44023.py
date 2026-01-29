# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified VGG-like model for style transfer (partial structure)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Layer 4 (0-based index)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Layer 9
        )
    
    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Capture outputs from layers relevant to style transfer (e.g., layers 0, 4, 9)
            if i in [0, 4, 9]:
                outputs.append(x)
        return outputs

def my_model_function():
    # Initialize the model with default weights
    return MyModel()

def GetInput():
    # Generate a random image tensor (batch_size=1, 3 channels, 224x224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

