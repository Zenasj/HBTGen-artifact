import torch

a = torch.tensor([[True,  True],
                  [False, True]])
print(a.t() == 0)