# torch.rand(1, 1, 48000, dtype=torch.float32)  # Inferred input shape for a dummy audio signal

import torch
import torch.nn as nn

class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TDNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, lengths=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ECAPA_TDNN(nn.Module):
    def __init__(self):
        super(ECAPA_TDNN, self).__init__()
        self.blocks = nn.Sequential(
            TDNNBlock(1, 512, 5, 1),
            TDNNBlock(512, 512, 3, 2),
            TDNNBlock(512, 512, 3, 3),
            TDNNBlock(512, 512, 1, 1),
            TDNNBlock(512, 1500, 1, 1)
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1500, 192)

    def forward(self, x, lengths=None):
        x = self.blocks(x)
        x = self.pooling(x).squeeze(-1)
        x = self.fc(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding_model = ECAPA_TDNN()

    def forward(self, x, lengths=None):
        return self.embedding_model(x, lengths)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 48000, dtype=torch.float32)

