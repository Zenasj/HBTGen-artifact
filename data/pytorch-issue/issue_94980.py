import torch

d = torch.tensor([[False,False, True, False],[False,False, False, True]]).to('mps')
e = d.nonzero()
e

tensor([[0, 2],
        [1, 3]], device='mps:0')

e[0].float()

tensor([0., 2.], device='mps:0')

e.float()[:,1]

tensor([2., 3.], device='mps:0')

e[:,1].float()

tensor([0., 1.], device='mps:0')