import torch

func_cls=torch.linalg.norm
# func_cls=torch.linalg.cond

a = torch.zeros(3, 3, requires_grad=True)
torch.linalg.matrix_norm(a).backward()
print(a.grad)

a.grad.zero_()
func_cls(a).backward()
print(a.grad)