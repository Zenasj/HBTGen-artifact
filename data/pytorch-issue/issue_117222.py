# python source code 
import torch
a = torch.randn(2,2,0,3)
print(a)
b = torch.sum(a, dim=2, keepdim=True)
print(b)

# error results
# a
tensor([], size=(2, 2, 0, 3))
# b
tensor([[[[0., 0., 0.]],

         [[0., 0., 0.]]],


        [[[0., 0., 0.]],

         [[0., 0., 0.]]]])
# Shouldn't 'b' be an empty tensor?