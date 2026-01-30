import torch.nn as nn

import numpy as np
import torch


def test(x):
    model = torch.nn.Linear(1000, 1010)

    x = x.cpu()
    model.cpu()
    y_cpu = model(x)

    x = x.cuda()
    model.cuda()
    y_gpu = model(x)
    print("cpu vs cuda: ", torch.allclose(y_cpu, y_gpu.cpu()))
    print("cpu vs cuda: ",
          np.max(np.abs(y_cpu.detach().cpu().numpy() - y_gpu.detach().cpu().numpy())))

    a, b = torch.testing._compare_tensors_internal(
            y_cpu, y_gpu.cpu(), rtol=0, atol=0, equal_nan=False)
    if not a:
        print(b)

x = torch.randn(32, 1024, 1000)
test(x)
x = torch.randn(32, 1024, 1000) + 100
test(x)