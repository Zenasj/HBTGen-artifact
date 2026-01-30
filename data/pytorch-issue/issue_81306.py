import torch
import torch.nn as nn

self.predictor = torch.load(model_path)
self.predictor.to('cuda')
self.predictor.eval()

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_dim):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
        self.act = nn.GELU()
        # self.dropout = nn.Dropout()
        self.output_dim = (input_dim - self.kernel_size) // self.stride + 1

    def forward(self, x, x_len):
        x = self.conv(x)
        x = self.act(x)
        # x = self.dropout(x)
        x_len = ((x_len - self.kernel_size) / self.stride + 1).int()
        return x, x_len

class GELU(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(input)
torch.nn.modules.activation.GELU = GELU