# torch.rand(B, 3, 256, 256, dtype=torch.float32)  # CycleGAN input shape (RGB images)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified ResNet-based generator architecture from CycleGAN
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            # Downsample
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            # Residual blocks (6 in original CycleGAN for 256x256)
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, kernel_size=3),
                nn.InstanceNorm2d(256),
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, kernel_size=3),
                nn.InstanceNorm2d(256),
            ),
            # Upsample
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()  # Returns a simplified CycleGAN generator instance

def GetInput():
    # Generates a random tensor matching the input shape (B, C, H, W)
    return torch.rand(2, 3, 256, 256, dtype=torch.float32)  # Batch size 2, RGB 256x256

