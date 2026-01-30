from torch.ao.quantization.fuse_modules import fuse_modules
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

m = M().eval()

modules_to_fuse = ['conv1', 'bn1', 'relu1']

print(f'before operator fusion: \n{m}\n')
fused_m = fuse_modules(m, modules_to_fuse)
print(f'after operator fusion: \n{fused_m}\n')