import torch
import torch.nn as nn
import torchvision.models as models

# torch.rand(B, 3, 32, W, dtype=torch.float32)  # Input shape: batch, channels=3, height=32, variable width
class MyModel(nn.Module):
    def __init__(self, num_classes=5991):
        super(MyModel, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=False).features  # Extract features
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # Maintain spatial width as sequence length
        self.fc = nn.Linear(1280, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)  # Apply over class dimension (dim=2)

    def forward(self, x):
        x = self.cnn(x)  # Output shape (N, 1280, H', W')
        x = self.pool(x)  # Output shape (N, 1280, 1, W')
        x = x.squeeze(2)  # Shape (N, 1280, W')
        x = x.permute(2, 0, 1)  # Shape (W', N, 1280)
        x = self.fc(x)  # (W', N, num_classes)
        return self.log_softmax(x)

def my_model_function():
    # Disable CuDNN to avoid non-deterministic CTC implementation causing NaN gradients
    torch.backends.cudnn.enabled = False
    return MyModel()

def GetInput():
    # Generate input with batch_size=2, 3 channels, height=32, variable width (e.g., 100)
    return torch.rand(2, 3, 32, 100, dtype=torch.float32)

