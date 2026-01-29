# torch.rand(B, 3, 28, 28, dtype=torch.float32)  # Assumed input shape for a simple CNN
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)  # Assuming global average pooling or flattening

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(x, 4)  # Simplified spatial reduction
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    # Initialize model with default weights and device handling
    model = MyModel()
    return model.to("cuda" if torch.cuda.is_available() else "cpu")  # Explicit device management

def GetInput():
    # Generate input matching the model's expected dimensions
    batch_size = 2
    return torch.rand(batch_size, 3, 28, 28, dtype=torch.float32).cuda()  # Assumes CUDA availability

