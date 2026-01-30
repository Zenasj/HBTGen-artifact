import torch
x = torch.ones([10, 13, 3, 3], dtype=torch.bfloat16)
x_trans = x.transpose(2, 3)
x_sum = torch.sum(x_trans, (0, 1, 2))
print(x_sum)