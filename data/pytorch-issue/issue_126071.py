# torch.rand(4, 10, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class FullyConnectedNet(nn.Module):
    def __init__(self, in_size, out_size, channels, dropout_prob):
        super(FullyConnectedNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.channels = channels
        self.dropout_prob = dropout_prob

        layers = []
        for i, channel in enumerate(channels):
            if i == 0:
                layers.append(nn.Linear(in_size, channel))
            else:
                layers.append(nn.Linear(channels[i-1], channel))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.Linear(channels[-1], out_size))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = FullyConnectedNet(10, 3, [8, 16], 0.15)

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(4, 10, dtype=torch.float32)

