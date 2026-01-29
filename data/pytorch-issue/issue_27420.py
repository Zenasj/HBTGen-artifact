# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

def BatchNorm2d(out_chan, momentum=0.1, eps=1e-3):
    return nn.SyncBatchNorm.convert_sync_batchnorm(
        nn.BatchNorm2d(out_chan, momentum=momentum, eps=eps)
    )

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn = BatchNorm2d(16)
        self.act = nn.ReLU(inplace=True)
        self.linear = nn.Linear(16, 1000)

    def forward(self, x):
        feat = self.act(self.bn(self.conv(x)))
        feat = torch.mean(feat, dim=(2, 3))
        logits = self.linear(feat)
        return logits

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

