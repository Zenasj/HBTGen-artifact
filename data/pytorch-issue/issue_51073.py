import torch
import torch.nn as nn
import torch.nn.functional as F

my_tensor = torch.rand(3,3)
print("Counting non-zeros of a random tensor")
print(torch.count_nonzero(my_tensor))

class MyMLP(nn.Module):
    """
    A simple MLP
    """
    def __init__(self):
        super(MyMLP, self).__init__()
        self.hidden1 = 512
        self.hidden2 = 512
        self.fc1 = nn.Linear(28 * 28, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

my_net = MyMLP()
print("Counting non-zeros of a weight tensor")
print(torch.count_nonzero(my_net.fc1.weight))