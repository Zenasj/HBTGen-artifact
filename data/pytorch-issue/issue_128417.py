import torch
import torchvision.transforms.v2 as transforms
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, img):
        w = max(img.shape[1], img.shape[2])  # Matches original code's logic (even if potentially incorrect)
        return transforms.CenterCrop(size=(w, w))(img)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor (B, C, H, W) compatible with the model's forward logic
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

