# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision import models

class ReNet(nn.Module):
    def __init__(self, n_input, n_units):
        super(ReNet, self).__init__()
        self.rnn = nn.GRU(n_input, n_units,
                          num_layers=1, batch_first=False,
                          bidirectional=True)

    def rnn_forward(self, x):
        b, n_height, n_width, n_filters = x.size()
        x = x.view(b * n_height, n_width, n_filters)
        x = x.permute(1, 0, 2)
        batch_size = x.size(1)
        hx = torch.zeros(2, batch_size, self.rnn.hidden_size, dtype=x.dtype, device=x.device)
        x, _ = self.rnn(x, hx)
        x = x.permute(1, 0, 2)
        x = x.view(b, n_height, n_width, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.rnn_forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-5])
        self.renet1 = ReNet(256, 50)

    def forward(self, x):
        x = self.cnn(x)
        x = self.renet1(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

