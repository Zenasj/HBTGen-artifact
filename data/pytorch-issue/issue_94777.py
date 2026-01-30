py
import torch

x = torch.tensor(3.)

def func(x):
    y = torch.log(x)
    x.log_()
    return y

print(func(x.clone()))
# tensor(1.0986)

print(torch.compile(func)(x.clone()))
# tensor(0.0940)

py
import torch

x = torch.tensor(3.)

def func(x):
    y = torch.log(x)
    # x.log_()
    return y

print(func(x.clone()))
# tensor(1.0986)

print(torch.compile(func)(x.clone()))
# tensor(1.0986)