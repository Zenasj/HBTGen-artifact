# torch.rand(B, 3, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimics the SAM model's image encoder and mask decoder structure
        # Image encoder outputs 256 channels (common in SAM architectures)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Simplified mask decoder with problematic shape unpacking
        self.mask_decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
    
    def forward(self, x):
        image_embeddings = self.image_encoder(x)
        # Reproduces the error-prone shape unpacking from the SAM model
        batch_size, channels, height, width = image_embeddings.shape
        # Dummy computation to simulate mask decoder logic
        pooled = image_embeddings.mean(dim=(2, 3))
        return self.mask_decoder(pooled)

def my_model_function():
    return MyModel()

def GetInput():
    # SAM expects images divisible by 32 (e.g., 1024x1024), batch size 1
    return torch.rand(1, 3, 1024, 1024, dtype=torch.float32).cuda()

