import torch

print(torch.__version__)    
x = torch.tensor([1.0,2.2],requires_grad=True)
print(x)  
y = x + 2  
print(y)  
print(y.grad_fn , x.grad_fn)