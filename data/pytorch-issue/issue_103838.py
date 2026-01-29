# torch.rand(B, 3, 48, 168, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False):
        super(MyModel, self).__init__()
        self.export = export
        self.feature = self.make_layers(cfg if cfg else [8,8,16,16,'M',32,32,'M',48,48,'M',64,128], batch_norm=True)
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0,1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1] if cfg else 128, num_classes, 1, 1)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3 if in_channels !=3 else 5, 
                                   padding=(1,1) if in_channels !=3 else 0,
                                   stride=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.feature(x)
        x = self.loc(x)
        x = self.newCnn(x)
        x = x.squeeze(2)  # b * C * width
        x = x.transpose(2,1)  # [width, b, C]
        return x

def my_model_function():
    # Using default configuration from issue's code
    return MyModel(cfg=[8,8,16,16,'M',32,32,'M',48,48,'M',64,128], num_classes=78)

def GetInput():
    # Matches input dimensions from issue's config.HEIGHT=48 and config.WIDTH=168
    return torch.randn(1, 3, 48, 168, dtype=torch.float32)

