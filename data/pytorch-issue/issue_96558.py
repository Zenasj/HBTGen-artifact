import torch
print(torch.flip(torch.Tensor([ [1, 2, 3] ]).to(torch.device('cpu')), [0]))

tensor([[1., 2., 3.]])

import torch
print(torch.flip(torch.Tensor([ [1, 2, 3] ]).to(torch.device('mps')), [0]))

tensor([[3., 2., 1.]], device='mps:0')

print(torch.flip(torch.Tensor([
    [ [1,2,3] ],
    [ [4,5,6] ]
    ]).to(torch.device('mps')), [1]))

tensor([[[6., 5., 4.]],
        [[3., 2., 1.]]], device='mps:0')