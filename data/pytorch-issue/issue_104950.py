import torch
a = torch.rand(2,3)
b = torch.linalg.qr(a)
print(b)
torch.return_types.linalg_qr(
Q=tensor([[-0.7239, -0.6899],
        [-0.6899,  0.7239]]),
R=tensor([[-0.9173, -1.0816, -1.1109],
        [ 0.0000, -0.0353, -0.1853]]))