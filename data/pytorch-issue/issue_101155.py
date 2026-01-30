import torch
import logging
import torch._dynamo

# torch._logging.set_logs(dynamo=logging.DEBUG, bytecode=True)
torch._dynamo.config.print_graph_breaks = True

import torch.nn as nn
import torch.nn.functional as F

class MyConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(MyConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )

    def forward(self, inputs, mask=None):
        outputs = super(MyConv1d, self).forward(inputs)
        return outputs

x = torch.randn(4, 4)

m = MyConv1d(4, 4, 4)
print(m(x))

opt_m = torch.compile(backend="eager")(m)
print(opt_m(x))