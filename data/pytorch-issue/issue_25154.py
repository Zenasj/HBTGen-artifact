import torch
import torch.nn as nn

class Bottleneck(nn.Module):

    def __init__(self,  downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
  
        return out

scripted_model = torch.jit.script(Bottleneck(nn.Conv2d(3, 32, 1))) # works
scripted_model = torch.jit.script(Bottleneck()) # Does not work