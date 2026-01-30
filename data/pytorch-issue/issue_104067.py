import torch.nn as nn

import torch
import io
from torch.utils.checkpoint import checkpoint


class A(torch.nn.Module):
    # A supported module.
    def __init__(self):
        super(A, self).__init__()
        self.l1 = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.l1(x)


class B(torch.nn.Module):
    # This module is not exportable to ONNX because it
    # uses gradient-checkpointing. However, its two sub-module's
    # are exportable, so ORTModule should be used to compute them.
    def __init__(self):
        super(B, self).__init__()
        self.l1 = torch.nn.Linear(2, 2)
        self.a = A()

    def forward(self, x):
        def custom():
            def custom_forward(x_):
                return self.a(x_)

            return custom_forward

        z = self.l1(checkpoint(custom(), x))
        return z


torch.onnx.export(
    B(),
    (torch.randn(2, 2),),
    io.BytesIO(),
    autograd_inlining=True
)