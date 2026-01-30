import torch

def func3(a, b):
    x = torch.nested.nested_tensor([a, b], layout=torch.jagged)
    x = x + 1.0
    x = x * 2.0
    x = x + 3.0
    x = torch.relu(x)
    return torch.abs(x)

a = torch.randn(3,3,device='cuda')
b = torch.randn(3,3,device='cuda')
compiled_func3 = torch.compile(func3)
c = compiled_func3(a, b)
print('nested value = ', nested.values())
print('c      value = ', c.values())