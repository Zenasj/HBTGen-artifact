import torch
import torch.nn as nn

# torch.rand(B, 3, 224, 224, dtype=torch.float32).cuda()  # Inferred input shape based on common CNN usage
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Simulate CUDA IPC stress
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 112 * 112, 10)  # Matches input shape after pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))  # Simulate multi-process tensor sharing pattern
        x = x.view(-1, 32 * 112 * 112)
        return self.fc(x)

def my_model_function():
    model = MyModel()
    model.cuda()  # Explicit CUDA placement to trigger IPC interactions
    return model

def GetInput():
    # Generate input that matches the model's expected dimensions and CUDA placement
    return torch.rand(2, 3, 224, 224, dtype=torch.float32).cuda()

