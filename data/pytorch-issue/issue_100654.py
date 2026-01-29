# torch.rand(B, 3, 112, 112, dtype=torch.float32)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    class IBasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.prelu = nn.PReLU(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn2(out)
            out = self.prelu(out)
            out = self.conv2(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        
        # Layer1
        self.layer1 = nn.Sequential(
            self.IBasicBlock(64, 64, stride=2),
            self.IBasicBlock(64, 64, stride=1)
        )
        # Layer2
        self.layer2 = nn.Sequential(
            self.IBasicBlock(64, 128, stride=2),
            self.IBasicBlock(128, 128, stride=1)
        )
        # Layer3
        self.layer3 = nn.Sequential(
            self.IBasicBlock(128, 256, stride=2),
            self.IBasicBlock(256, 256, stride=1)
        )
        # Layer4
        self.layer4 = nn.Sequential(
            self.IBasicBlock(256, 512, stride=2),
            self.IBasicBlock(512, 512, stride=1)
        )
        
        self.dropout = nn.Dropout(p=0, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512)  # Included per model structure but unused in forward
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # Omit features layer as per printed forward path

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 112, 112, dtype=torch.float32)

