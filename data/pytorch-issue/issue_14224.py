import torch
import torch.nn as nn

torch.nn.functional.log_softmax(torch.arange(9), dim=0)

t1 = torch.tensor([5, 2, 87], dtype=torch.float16)
r1 = torch.tensor([8, 32, -27], dtype=torch.float16)
t2 = torch.tensor([756, 15, 327], dtype=torch.float16)