import torch
import torch.nn as nn
import torch.nn.functional as F

class TestMod(nn.Module):
    def __init__(self, size, maxint=10):
        super().__init__()
        self.maxint = maxint
        self.weight = torch.randint(0, maxint, [size])

    def forward(self, x):
        return F.one_hot(self.weight, num_classes=self.maxint)
test = TestMod(3)
torch.onnx.export(test, test.weight, "test.onnx", verbose=True)