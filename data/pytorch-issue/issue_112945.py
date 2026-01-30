from torch.ao.quantization.fuse_modules import fuse_modules
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 64)  # Linear layer with appropriate input and output dimensions
        self.bn1 = nn.BatchNorm1d(64)  # Use BatchNorm1d for 1D input
        self.fc2 = nn.Linear(64, 64)  # Linear layer with appropriate input and output dimensions
        self.bn2 = nn.BatchNorm1d(64)  # Use BatchNorm1d for 1D input

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


m = M().eval()

modules_to_fuse = ['fc1', 'bn1']

print(m)
fused_m = fuse_modules(m, modules_to_fuse)
print(fused_m)