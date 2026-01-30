input_t = torch.tensor([0.j], requires_grad=True)
sqaure_root = input_t.pow(1/2)
loss = sqaure_root.sum() # loss is tensor(0.+0.j)
loss.backward()
print(input_t.grad)

import torch
print(torch.tensor([0.j]).pow(-1/2))
print(torch.tensor([0.]).pow(-1/2))