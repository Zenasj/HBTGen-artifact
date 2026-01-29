# torch.rand(B*N, 4, dtype=torch.float32)  # Assuming 3 (xyz) + 1 (radius) channels
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_channels=4, hidden=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Linear(hidden, hidden*2), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Linear(hidden*2, hidden*4), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Linear(hidden*4, hidden*8), nn.ReLU())
        self.enc5 = nn.Sequential(nn.Linear(hidden*8, hidden*16), nn.ReLU())

        self.dec5 = nn.Sequential(nn.Linear(hidden*16, hidden*8), nn.ReLU())
        self.dec4 = nn.Sequential(nn.Linear(hidden*8, hidden*4), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Linear(hidden*4, hidden*2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.cls = nn.Linear(hidden, 10)  # Placeholder output size

    def forward(self, pc):
        n, c = pc.shape
        p0 = pc[:, :3].contiguous()  # XYZ coordinates
        x0 = pc  # Full input features
        o0 = torch.tensor([n], dtype=torch.int32).cuda()  # Batch offset (assuming single batch)

        # Encoder pathway
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Decoder pathway with skip connections
        x5_dec = self.dec5(x5)
        x4_dec = self.dec4(x5_dec + x4)
        x3_dec = self.dec3(x4_dec + x3)
        x2_dec = self.dec2(x3_dec + x2)
        x1_dec = self.dec1(x2_dec + x1)
        x = self.cls(x1_dec)

        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    N = 1024  # Points per batch
    C = 4  # 3 (xyz) + 1 (radius)
    return torch.rand(B*N, C, dtype=torch.float32).cuda()  # Match CUDA usage in original code

