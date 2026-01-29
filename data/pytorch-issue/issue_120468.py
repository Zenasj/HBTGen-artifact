import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed input shape (1, 3, 224, 224)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 222 * 222, 10)  # After 3x3 conv, input becomes 222x222
        
        # Register forward hooks to capture module signatures (avoids monkey-patching)
        self.conv.register_forward_hook(self._capture_signature)
        self.relu.register_forward_hook(self._capture_signature)

    def _capture_signature(self, module, input, output):
        # Dummy hook implementation (placeholder for signature tracking logic)
        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Returns an instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

