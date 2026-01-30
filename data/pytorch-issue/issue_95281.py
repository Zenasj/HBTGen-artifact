import torch
torch.manual_seed(0)

batch=9
# 8
# 2
# 1
func_cls=torch.logdet
A = torch.tensor([[1e-40, 1e-40], [1e-40, 1e-40]])


B = torch.tile(A, [batch,1,1])
# print(B)
x = func_cls(B).cpu().detach().numpy()

D = torch.tile(A, [batch,1,1]).cuda()
# print(D)
y = func_cls(D).cpu().detach().numpy()
print(f'Cuda result: {y}')
print(f'CPU result: {x}')