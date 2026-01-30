import torch
import torch._dynamo
import torch._inductor.config

torch._inductor.config.cpp_wrapper = True

x = torch.rand((16, 16, 16), device='cuda')

def fn(x):
    y = x + 4
    z = torch.fft.fftn(y)
    return z

torch._dynamo.optimize("inductor")(fn)(x)