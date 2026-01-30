import torch
from torch.func import vmap
import numpy as np

a = torch.randn((2,10))
b = torch.randn((2,20))

def np_convolve(x, y):
    x, y = x.numpy(), y.numpy()
    # Here could be some code only available on numpy array ...
    z = np.convolve(x, y)
    return torch.from_numpy(z)

@torch.compile
def vmap_convolve(x, y):
    return vmap(np_convolve)(x,y)

def poormans_vmap(x, y):
    x, y = x.numpy(), y.numpy()
    z = np.stack([np.convolve(x1, y1) for x1, y1 in zip(x, y)])
    return torch.from_numpy(z)

torch.testing.assert_close(vmap_convolve(a, b), poormans_vmap(a, b))

import torch
from torch.func import vmap

a = torch.randn((2,10))
b = torch.randn((2,20))

def np_convolve(x, y):
    import torch._numpy as np
    x, y = np.asarray(x), np.asarray(y)
    # Here could be some code only available on numpy array ...
    z = np.convolve(x, y)
    return z.tensor

def vmap_convolve(x, y):
    return vmap(np_convolve)(x,y)

def poormans_vmap(x, y):
    import numpy as np
    x, y = x.numpy(), y.numpy()
    result = [np.convolve(x1, y1) for x1, y1 in zip(x, y)]
    z = np.stack(result)
    return torch.from_numpy(z)

torch.testing.assert_close(vmap_convolve(a, b), poormans_vmap(a, b))