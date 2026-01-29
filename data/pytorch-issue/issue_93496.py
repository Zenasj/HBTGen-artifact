# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mock Vision Transformer architecture for image input
        self.patch_embedding = nn.Conv2d(3, 768, kernel_size=16, stride=16)  # 224/16=14 → 14x14 patches
        self.positional_enc = nn.Parameter(torch.randn(196, 768))  # 14x14=196 patches
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.fc = nn.Linear(768, 1000)  # Example output layer

    def forward(self, x):
        x = self.patch_embedding(x)  # (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
        x += self.positional_enc
        x = self.transformer(x)
        x = self.fc(x.mean(dim=1))  # Global average pooling
        return x

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random image tensor compatible with MyModel's input requirements
    B = 16  # Example batch size
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)

