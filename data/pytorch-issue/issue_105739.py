import torch
def loss_func(x,y):
    return torch.mean((x-y)**2)

x = torch.rand(10,1)
y = torch.rand(10)

print(loss_func(x,y) == loss_func(torch.squeeze(x),y))
print(torch.abs(loss_func(x,y)-loss_func(torch.squeeze(x),y))>0.01)

import torch
x = torch.tensor([10, 20, 30, 40, 50]).reshape(5, 1)
y = torch.tensor([1, 2, 3, 4, 5])
a = x - y 
b = torch.squeeze(x) - y