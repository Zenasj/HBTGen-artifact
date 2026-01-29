# torch.rand(B, 1, 32, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, inputs, outputs):
        super(Layer, self).__init__()
        k = 3
        p = (k - 1) // 2
        s = 1
        self.layer = nn.Sequential(
            nn.Conv3d(inputs, outputs, kernel_size=k, padding=p, stride=s),
            nn.ReLU(inplace=True),
            nn.Conv3d(outputs, outputs, kernel_size=k, padding=p, stride=s),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoding_layers = nn.ModuleList()
        self.encoding_layers.append(Layer(1, 32))
        self.encoding_layers.append(Layer(32, 64))
        self.encoding_layers.append(Layer(64, 128))
        self.encoding_layers.append(Layer(128, 256))

        self.decoding_layers = nn.ModuleList()
        self.decoding_layers.append(Layer(256, 128))
        self.decoding_layers.append(Layer(128, 64))
        self.decoding_layers.append(Layer(64, 32))

        self.downsamples = nn.ModuleList()
        self.downsamples.append(nn.MaxPool3d(2))
        self.downsamples.append(nn.MaxPool3d(2))
        self.downsamples.append(nn.MaxPool3d(2))

        self.upsamples = nn.ModuleList()
        self.upsamples.append(nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2))
        self.upsamples.append(nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2))
        self.upsamples.append(nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2))

        self.map_to_output = nn.Conv3d(32, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.encoding_layers[0](x)
        x2 = self.encoding_layers[1](self.downsamples[0](x1))
        x3 = self.encoding_layers[2](self.downsamples[1](x2))
        x = self.encoding_layers[3](self.downsamples[2](x3))

        x = self.upsamples[0](x)
        x = self.decoding_layers[0](torch.cat([x3, x], dim=1))

        x = self.upsamples[1](x)
        x = self.decoding_layers[1](torch.cat([x2, x], dim=1))

        x = self.upsamples[2](x)
        x = self.decoding_layers[2](torch.cat([x1, x], dim=1))

        return self.map_to_output(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 32, 32, 32, dtype=torch.float32)

