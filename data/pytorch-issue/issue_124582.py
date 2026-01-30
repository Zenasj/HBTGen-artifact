import torch.nn as nn

import torch
from torch import nn


def func(x):
    return torch.gather(x, 0, torch.tensor(0, device=x.device))


class Model(nn.Module):

    def forward(self, x):
        return torch.gather(x, 0, torch.tensor(0, device=x.device))


if __name__ == "__main__":
    args = torch.randn(1, requires_grad=False).cuda()
    model = Model().cuda()

    # model and func fail with the same error
    model = torch.cuda.make_graphed_callables(model, (args, ))
    func = torch.cuda.make_graphed_callables(func, (args, ))