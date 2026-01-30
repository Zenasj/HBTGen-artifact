import random

import numpy as np
import torch
import torch.nn as nn


class SymbolNet(nn.Module):
    def __init__(self):
        super(SymbolNet, self).__init__()
        self.m3 = nn.Linear(in_features=1, out_features=43, bias=True)
        self.m4 = nn.MaxPool2d(kernel_size=(2, 42), stride=2, padding=0, dilation=1, ceil_mode=False)
        self.m5 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.m3(x)
        x = self.m4(x)
        x = self.m5(x)
        return x


model = SymbolNet()

inp = np.random.rand(24, 1, 4, 1).astype('float32')

m_out = model(torch.from_numpy(inp).to('cpu'))
m_out = [v.cpu().detach() for v in m_out]  # torch2numpy
m_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in m_out]

# Compile the model
opt = torch.jit.trace(model.eval(), torch.from_numpy(inp).to('cpu'))
# Compiled run
opt_out = opt(torch.from_numpy(inp).to('cpu'))
opt_out = [v.cpu().detach() for v in opt_out]
opt_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in opt_out]

# Differential testing
for i, (l, r) in enumerate(zip(m_out, opt_out)):
    np.testing.assert_allclose(l, r, rtol=1e-2, atol=1e-3, err_msg=f"Result mismatch @ index {i}")

# AssertionError:
# Not equal to tolerance rtol=0.01, atol=0.001
# Result mismatch @ index 0
# Mismatched elements: 2 / 2 (100%)
# Max absolute difference among violations: 2.5436974
# Max relative difference among violations: 2.408335
#  ACTUAL: array([[[-0.560976],
#         [-1.487492]]], dtype=float32)
#  DESIRED: array([[[1.169456],
#         [1.056206]]], dtype=float32)