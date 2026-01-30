import torch

from torch import Generator, randn

g = Generator(device='mps')
# RuntimeError: Device type MPS is not supported for torch.Generator() api.

# lol okay fine we'll make it on-CPU

g = Generator(device='cpu')
g.manual_seed(42)
randn(3, generator=g, device='cpu')
# tensor([0.3367, 0.1288, 0.2345])

g.manual_seed(42)
randn(3, generator=g, device='cpu')
# tensor([0.3367, 0.1288, 0.2345])
# cool, it's deterministic if everything's done on-CPU.

# now what happens if we try a CPU Generator with MPS randn()?

g.manual_seed(42)
randn(3, generator=g, device='mps')
# tensor([-1.0727, -0.2386,  0.5970], device='mps:0')

g.manual_seed(42)
randn(3, generator=g, device='mps')
# tensor([-0.4420,  0.6986, -0.1786], device='mps:0')
# CPU Generator with MPS randn() is non-deterministic