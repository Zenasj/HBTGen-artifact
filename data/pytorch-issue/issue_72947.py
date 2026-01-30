import torch
t = torch.rand(10,  dtype=torch.complex64)
t.imag = 0  # Segmentation fault (core dumped)
t.real = 0  # Segmentation fault (core dumped)