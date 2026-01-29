# torch.rand(B, 3, 300, 300, dtype=torch.float32)  # e.g., (1, 3, 300, 300)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.backbone1 = nn.Sequential(
            nn.Conv2d(3, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.backbone2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.backbone3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        self.loc = nn.ModuleList([nn.Conv2d(512, 16, 3, padding=1) for _ in range(3)])  # 4*4 output channels
        self.conf = nn.ModuleList([nn.Conv2d(512, 8, 3, padding=1) for _ in range(3)])  # 2*4 output channels
        self.num_classes = 2

    def forward(self, x):
        source1 = self.backbone1(x)
        source2 = self.backbone2(source1)
        source3 = self.backbone3(source2)
        sources = [source1, source2, source3]
        loc = []
        conf = []
        for x_s, l, c in zip(sources, self.loc, self.conf):
            # Duplicate convolutions due to repeated calls in the same line (problematic pattern)
            loc.append(l(x_s).permute(0,2,3,1).contiguous().view(l(x_s).size(0), -1,4))
            conf.append(c(x_s).permute(0,2,3,1).contiguous().view(c(x_s).size(0), -1, self.num_classes))
        loc = torch.cat(loc, 1)
        conf = torch.cat(conf, 1)
        return loc, conf

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 300, 300, dtype=torch.float32)

