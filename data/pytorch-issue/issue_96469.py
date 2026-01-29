# torch.rand(B, 1, H, W, dtype=torch.float32, device='cuda')  # B, H, W are symbolic dimensions
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)).cuda()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)).cuda()
        self.dropout1 = nn.Dropout(p=0.25, inplace=False)
        self.fc1 = nn.Linear(in_features=9216, out_features=128, bias=True).cuda()
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(in_features=128, out_features=10, bias=True).cuda()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random tensor input with symbolic dimensions B, H, W
    B, H, W = 64, 28, 28  # Example values for B, H, W
    return torch.rand(B, 1, H, W, dtype=torch.float32, device='cuda')

