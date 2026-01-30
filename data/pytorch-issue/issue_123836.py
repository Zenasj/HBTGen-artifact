import torch

M = torch.zeros(3,3, device='cpu')

M[1,1] = torch.tensor([3.], device='mps')
M[-1,-1] = torch.tensor([-3.], device='mps')
M[0,1] = torch.tensor([32.], device='mps')

print(M)

"""
tensor([[32.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
"""