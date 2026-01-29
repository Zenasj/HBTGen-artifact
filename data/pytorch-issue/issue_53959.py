# torch.rand(B, 3, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN for segmentation task matching input shape
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)  # Output 1 channel mask

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Random input matching HuBMAPDataset's image dimensions (assumed 256x256)
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

