import torch

t = torch.zeros((5, 3, 6), dtype=torch.complex128, device="meta")

def func(a):
    print("a:")
    print("size:", a.shape)
    print("stride:", a.stride())
    b = torch._fft_c2c(a, [1], 1, True)
    print("b:")
    print("size:", b.shape)
    print("stride:", b.stride())
    c = torch._conj_physical(b)
    print("c:")
    print("size:", c.shape)
    print("stride:", c.stride())
    return c

func(t)

import torch

t = torch.zeros((5, 3, 6), dtype=torch.complex128, device='cuda')

def func(a):
    print("a:")
    print("size:", a.shape)
    print("stride:", a.stride())
    b = torch.ops.aten._fft_c2c.default(a, [1], 1, True)
    print("b:")
    print("size:", b.shape)
    print("stride:", b.stride())
    c = torch.ops.aten._conj_physical.default(b)
    print("c:")
    print("size:", c.shape)
    print("stride:", c.stride())
    return c

func(t)

c = torch.ops.aten._conj_physical(b, out=torch.empty_like(b))

import torch

t = torch.zeros((5, 3, 6), dtype=torch.complex128, device="meta")

def func(a):
    b = torch._fft_c2c(a, [1], 1, True)
    print(torch.ops.aten._conj_physical(b, out=torch.empty_strided((5, 3, 6), (1, 5, 15), dtype=torch.complex128, device='meta')).stride())
    print(torch.conj_physical(b, out=torch.empty_strided((5, 3, 6), (1, 5, 15), dtype=torch.complex128, device='meta')).stride())

func(t)

(1, 5, 15)
(18, 1, 3)