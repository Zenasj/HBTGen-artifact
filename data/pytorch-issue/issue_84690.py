import os
import torch
import torch.nn as nn

# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Example input shape (batch, channels, height, width)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 10)  # Example output layer

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set visible GPU to physical device 1 (as in the issue)
    model = MyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

def GetInput():
    # Generate input matching the model's expected dimensions
    return torch.rand(8, 3, 224, 224, dtype=torch.float32).to("cuda")  # Batch size 8

