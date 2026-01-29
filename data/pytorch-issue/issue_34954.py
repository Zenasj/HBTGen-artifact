# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple model to generate embeddings and label images
        self.fc = nn.Linear(3 * 32 * 32, 2)  # Example embedding layer
    
    def forward(self, x):
        # Compute embeddings
        embeddings = self.fc(x.view(x.size(0), -1))
        # Return embeddings and label images (permuted to HWC format)
        return embeddings, x.permute(0, 2, 3, 1)  # Convert (B,C,H,W) → (B,H,W,C)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random image tensor with shape (batch, channels, height, width)
    return torch.rand(32, 3, 32, 32)  # Example input: 32 samples of 3x32x32 images

