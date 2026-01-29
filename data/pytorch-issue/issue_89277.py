# torch.rand(B, 512, H, dtype=torch.float32)
import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, kernel_size=3, dilation=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2
        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [128 * 2, 64 * 2, 32 * 2, 16 * 2]
        upsample_kernels = [1, 2, 2, 2]
        in_channels = 512  # Initial input channels from encoder
        self.upsamples = nn.ModuleList()
        for k in range(4):
            upsample = nn.Upsample(scale_factor=upsample_kernels[k])
            conv = ConvBNReLU(
                in_channels=in_channels if k == 0 else filters[k - 1],
                out_channels=filters[k],
                kernel_size=3,  # Inferred from default in ConvBNReLU
                activation='relu',
            )
            self.upsamples.append(nn.Sequential(upsample, conv))
            in_channels = filters[k]  # Track channel count for next layer

    def forward(self, x):
        for upsample in self.upsamples:
            x = upsample(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (B, C, H) matching initial 512 channels and arbitrary length (4 as a minimal example)
    return torch.rand(1, 512, 4, dtype=torch.float32)

