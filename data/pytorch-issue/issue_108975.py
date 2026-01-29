# torch.rand(B, 3, 448, 448, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, size, ch_in, ch_out, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, size, stride, padding, bias=False)
        self.batchNorm = nn.BatchNorm2d(ch_out)
        self.lrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.lrelu(self.batchNorm(self.conv(x)))

class MyModel(nn.Module):
    def __init__(self, classes=1000, init_weight=False):
        super(MyModel, self).__init__()
        self.darkNet = nn.Sequential(
            ConvBlock(7, 3, 64, 2, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(3, 64, 192, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(1, 192, 128),
            ConvBlock(3, 128, 256, padding=1),
            ConvBlock(1, 256, 256),
            ConvBlock(3, 256, 512, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(1, 512, 256),
            ConvBlock(3, 256, 512, padding=1),
            ConvBlock(1, 512, 256),
            ConvBlock(3, 256, 512, padding=1),
            ConvBlock(1, 512, 256),
            ConvBlock(3, 256, 512, padding=1),
            ConvBlock(1, 512, 256),
            ConvBlock(3, 256, 512, padding=1),
            ConvBlock(1, 512, 512),
            ConvBlock(3, 512, 1024, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(1, 1024, 512),
            ConvBlock(3, 512, 1024, padding=1),
            ConvBlock(1, 1024, 512),
            ConvBlock(3, 512, 1024, padding=1)
        )
        self.classifier = nn.Sequential(
            *self.darkNet,
            GlobalAvgPool2d(),
            nn.Linear(1024, classes)
        )
        
        if init_weight:
            self._initialize_weights()
    
    def forward(self, x):
        return self.classifier(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ConvBlock):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.batchNorm.weight, 1)
                nn.init.constant_(m.batchNorm.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 448, 448, dtype=torch.float32)

