py
import torch
import torch.nn as nn

torch.manual_seed(420)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 32 * 8 * 8)
        return x
input_tensor = torch.rand(1, 3, 32, 32)

func = Net().to('cpu')

func.train(False)
with torch.no_grad():
    res1 = func(input_tensor)
    print(res1)
    res2 = torch.compile(func)(input_tensor)
    print(res2)

py
import torch
import torch.nn as nn

torch.manual_seed(420)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 32 * 8 * 8)
        return x
input_tensor = torch.rand(1, 3, 32, 32)

func = Net().to('cpu')

func.train(True)
with torch.no_grad():
    res1 = func(input_tensor)
    print(res1)
    res2 = torch.compile(func)(input_tensor)
    print(res2)

tensor([[0.0000, 0.0000, 0.0000,  ..., 0.1241, 0.1636, 0.1719],
        [0.0134, 0.0000, 0.1067,  ..., 0.0195, 0.0660, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0467, 0.0545, 0.1448],
        ...,
        [0.0000, 0.0031, 0.0000,  ..., 0.0160, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0370, 0.0416, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0958, 0.1326, 0.0726]])
tensor([[0.0000, 0.0000, 0.0000,  ..., 0.1241, 0.1636, 0.1719],
        [0.0134, 0.0000, 0.1067,  ..., 0.0195, 0.0660, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0467, 0.0545, 0.1448],
        ...,
        [0.0000, 0.0031, 0.0000,  ..., 0.0160, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0370, 0.0416, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0958, 0.1326, 0.0726]])

py
import torch
import torch.nn as nn

set_seed(420)


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 8 * 8, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 8 * 8)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

input_tensor = torch.randn((1, 3, 32, 32))
func = MyModel().to('cpu')

func.train(False)
with torch.no_grad():
    # without optimization
    res1 = func(input_tensor)
    print(res1)

    # with optimization
    # it triggered the fuse_unary optimization
    try:
        fn = torch.compile(func)
        res2 = fn(input_tensor)
    except Exception as e:
        print(e)