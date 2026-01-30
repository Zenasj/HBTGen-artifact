import torch
import torch.nn as nn

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)
        self.buffer = torch.zeros([1, self.in_channels, self.left_padding])

    def forward(self, input, end_flag):
        x = torch.cat((self.buffer, input), 2)

        self.buffer = x[:, :, input.shape[2]:]
        self.buffer = self.buffer * end_flag[0]

        return super(CausalConv1d, self).forward(x)