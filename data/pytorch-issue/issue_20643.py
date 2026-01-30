import torch.nn as nn
import torch.nn.functional as F

class test(nn.Module):

    def __init__(self):
        super(test, self).__init__()

    def forward(self, x):
        return F.softmax(x, dim=1)

nn.LogSoftmax(dim=1)