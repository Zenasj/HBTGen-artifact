# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified SegResNet structure mimicking the reverse operation context
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv3d(4, 16, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.Dropout3d(0.2))
        ])
        self.decoder_first_layer = nn.Conv3d(128, 3, 1)  # Dummy output layer

    def forward(self, x):
        down_x = []
        for block in self.enc_blocks:
            x = block(x)
            down_x.append(x)
        # Replace list.reverse() with slicing to avoid unsupported operation
        reversed_down = down_x[::-1]
        # Dummy decoder step using reversed features (first element here)
        return self.decoder_first_layer(reversed_down[0])

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 4, 224, 224, 128, dtype=torch.float32).cuda()

