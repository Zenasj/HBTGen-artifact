import torch
import torch.nn as nn
import numpy as np

class WalshHadamard(torch.nn.Module):
    def __init__(self, sample_x):
        super().__init__()

        h_dim = int(sample_x.shape[0])
        assert _is_power_of_two(h_dim)
        self.h_dim_exp = int(np.log(h_dim) / np.log(2))
        self.working_shape = [1] + ([2] * self.h_dim_exp) + [1]

    def forward(self, x):
        result = x.view(self.working_shape)

        for i in range(self.h_dim_exp):
            dim = i + 1
            arrs = torch.chunk(result, 2, dim=dim)
            result = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

        return result.view(x.shape)

self.walsh_hadamard = torch.cuda.make_graphed_callables(WalshHadamard(mul_1), (mul_1,))