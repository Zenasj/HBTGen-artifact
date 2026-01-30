import torch.nn as nn

import torch
import traceback

print(torch.__version__)

def test(f):
    try:
        _test(f)
    except:
        traceback.print_exc()

def _test(f):
    x = torch.randn(1, 2, 8, 8, dtype=torch.float, device='cuda')\
        .to(memory_format=torch.channels_last)
    x.requires_grad_()

    # y = torch.nn.functional.relu(x)
    y = f(x)
    g = torch.randn_like(y)

    assert y.is_contiguous(memory_format=torch.channels_last)
    assert g.is_contiguous(memory_format=torch.channels_last)
    y.backward(g)
    assert x.grad.is_contiguous(memory_format=torch.channels_last)

    torch.cuda.synchronize()


if __name__ == '__main__':
    test(lambda x: torch.nn.functional.max_pool2d(x, 3))
    test(torch.nn.functional.relu)
    test(lambda x: torch.nn.functional.interpolate(x, 16))