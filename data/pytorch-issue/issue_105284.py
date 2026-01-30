import torch
x = torch.tensor([1, 2, 3, 5, 6], device="mps", dtype=torch.int16)
x.unsqueeze(-1).expand(x.shape + (3,)).cumsum(-1)

import torch
x = torch.tensor([1, 2, 3, 5, 6], device="mps", dtype=torch.int16)
x.unsqueeze(-1).expand(x.shape + (3,)).logit(-1)