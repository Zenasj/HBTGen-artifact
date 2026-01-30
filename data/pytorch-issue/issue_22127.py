import torch
import torch.nn as nn
import torch.nn.functional as F

class TestLeakModel(nn.Module):
    def __init__(self):
        super(TestLeakModel,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=16, kernel_size=3, padding=1)   # WEIGHTs ARE INITED
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                               padding=1)  # WEIGHTs ARE INITED
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)  # WEIGHTs ARE INITED
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                               padding=1)  # WEIGHTs ARE INITED
        self.fc1 = nn.Linear(in_features=64*16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=128)
        self.fc6 = nn.Linear(in_features=128, out_features=7)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = F.softmax(x)
        return x

def test():
    x = torch.randn((1, 18, 8, 8))
    import gc
    for i in range(10000000):
        # gc.collect()
        with torch.no_grad():
            testmodel.forward(x)