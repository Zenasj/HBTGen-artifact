import torch.nn as nn

py
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_1 = torch.nn.Linear(64, 10000)
        self.dense_2 = torch.nn.Linear(10000, 64)

    def forward(self, input):
        input = torch.relu(self.dense_1.forward(input))
        input = self.dense_2.forward(input)
        return input

outputs = []
net = Net()
net.eval()
while True:
    input = torch.randn(1, 64)
    output = net.forward(input)
    outputs.append(output)