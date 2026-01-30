import torch.nn as nn

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16, min_channels=8):
        super(SEModule, self).__init__()
        reduction_channels = max(channels // reduction, min_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

import torch
import timm  # pip install

model = timm.create_model('seresnet50', pretrained=True)
model = model.cuda().to(memory_format=torch.channels_last)
model.eval()

with torch.no_grad():
    for _ in range(1000):
        x = torch.randn(2, 3, 224, 224).cuda().to(memory_format=torch.channels_last) * 10
        with torch.cuda.amp.autocast():
            out = model(x)
            if not torch.isfinite(out).all():
                print('ERROR')
                break