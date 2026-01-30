import torch.nn as nn

import torch
import torchvision
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        return x

myNetwork = Network()

data = torch.Tensor([3]).long()
data = torch.autograd.Variable(data)
y_ = myNetwork(data)