import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._inductor import config

config.fallback_random = True  # here is the trigger condition


class Model(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dropout = eval(f"nn.Dropout{dim}d(p=0.5)")

    def forward(self, x):
        torch.manual_seed(0)
        x = self.dropout(x)
        return x


def run_test(dim, device):
    torch.manual_seed(0)
    shape = [1, 3] + [256] * dim
    inputs = torch.randn(*shape).to(device)

    model = Model(dim).to(device)
    output = model(inputs)
    c_model = torch.compile(model)
    c_output = c_model(inputs)

    print(torch.allclose(output, c_output, 1.3e-6, 1e-5))
    print(torch.max(torch.abs(output - c_output)))


run_test(dim=1, device='cpu')
run_test(dim=2, device='cpu')
run_test(dim=3, device='cpu')
run_test(dim=1, device='cuda')
run_test(dim=2, device='cuda')
run_test(dim=3, device='cuda')