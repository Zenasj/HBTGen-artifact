import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(torch.jit.ScriptModule):

    def __init__(self):
        super(Net,self).__init__()

        kernel_size = 3
        stride = 1
        filters = 1

        self.conv1 = nn.Conv2d(1, filters, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(filters)

    @torch.jit.script_method
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        return x


m = Net()

m.eval()
with torch.no_grad():
    output = m(torch.zeros((1,1,10,10)))