# torch.rand(B, 3, 24, 28, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(8 * 24 * 28, 2)  # Assuming output after conv is (N, 8, 24, 28)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        theta_i = self.fc(x)

        scale_factors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=x.dtype, device=x.device)
        theta_i_0 = theta_i[0]  # Assuming batch size 1
        theta_tmp = torch.stack([theta_i_0[0], theta_i_0[1]]).unsqueeze(1)

        # Apply the fix with transposes to resolve TorchScript type inference issue
        scale_t = scale_factors.transpose(0, 1)
        theta_t = theta_tmp.transpose(0, 1)
        combined = torch.cat([scale_t, theta_t], dim=0)
        combined = combined.transpose(0, 1)
        theta = combined.unsqueeze(0)

        return theta

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 24, 28, dtype=torch.float32)

