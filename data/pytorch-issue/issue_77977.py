import torch as pt

x = pt.ones(3, device='mps')  # sizes 1 and 2 work, but result is empty tensor
m = pt.ones_like(x).bool()  # pt.zeros_like works
x[m]  # crash