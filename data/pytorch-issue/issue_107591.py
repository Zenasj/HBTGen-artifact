import torch

x = torch.ones(2, 32, 64, 64).to("cuda")
n_repeat = torch.tensor(4).to(torch.int32)

res = x.repeat_interleave(n_repeat, 0)

import torch

x = torch.ones(2, 32, 64, 64).to("cuda")
n_repeat = torch.tensor(4).to(torch.int32)

res = x.repeat(n_repeat, 1, 1, 1)