import torch

A = torch.ones(5, 2, dtype=torch.double)
b = torch.rand(5) 
tensor([0.5433, 0.1091, 0.3717, 0.8436, 0.4544])


# Single-column assignment works fine
A[:, 1] = b
A
tensor([[1.0000, 0.5433],
        [1.0000, 0.1091],
        [1.0000, 0.3717],
        [1.0000, 0.8436],
        [1.0000, 0.4544]], dtype=torch.float64)


# Multi-column assignment breaks, presumably b/c of type promotion
A[:, [1]] = b.unsqueeze(-1)
A
tensor([[ 1.0000e+00,  1.1426e-10],
        [ 1.0000e+00,  1.8074e-06],
        [ 1.0000e+00,  1.4630e-03],
        [ 1.0000e+00,  1.1764e-05],
        [ 1.0000e+00, 5.2145e-315]], dtype=torch.float64)


# Multi-column assignment works if no type promotion is involved
A[:, [1]] = b.double().unsqueeze(-1)
A
tensor([[1.0000, 0.5433],
        [1.0000, 0.1091],
        [1.0000, 0.3717],
        [1.0000, 0.8436],
        [1.0000, 0.4544]], dtype=torch.float64)