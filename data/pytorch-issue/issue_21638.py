import torch
import numpy as np


axes = (1, 3)

r = torch.tensor([[[[-0.47391583903785255982,  0.99425932417625995097]],
                   [[-0.16560788865645595380, -0.94544471859987488926]]],
                  [[[ 0.86797840599953324237, -0.07809049467944455258]],
                   [[-1.15369110156186338578, -0.83403423341840843275]]]],
                 dtype=torch.float32, device='cuda:0')

x = r[:, :, :, ::2]

a = np.array(x.cpu(), copy=False)

expected = np.linalg.norm(a, "nuc", axis=axes)

ans = torch.norm(x, "nuc", dim=axes)

print("%r\n%r\n" % (ans, expected))

tensor([[0.5020],
        [1.4437]], device='cuda:0')
array([[0.50201815],
       [1.4437416 ]], dtype=float32)