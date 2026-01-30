import torch
x=torch.randn(1, requires_grad=True)
w=torch.tensor(float("nan"))
z=(x*w).nan_to_num(0)
print(z) # tensor([0.], grad_fn=<NanToNumBackward0>)
z.backward()
print(x.grad) # tensor([nan])