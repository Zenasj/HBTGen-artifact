import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def forward(self, x):
        return x

net = NeuralNetwork()
criterion = CrossEntropyLoss()
N_EPOCHS = 10

for e in range(N_EPOCHS):
    print(e)
    for x, y in dataLoader: # `dataLoader` is a DataLoader object
        logit = net(x)

        loss = criterion(logit, y)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

net = NeuralNetwork()#.to(device)
x = torch.rand(1, 28, 28)
net(x)

a = net(x)
print(a)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
net.to(device)