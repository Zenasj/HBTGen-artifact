import torch
from torch import Tensor
t = Tensor([[-1,2]])
ttr = t.transpose(0,1)
print(f" t is: \n{t}")
print(f" ttr is: \n{ttr}")
print("-"*20)

t[0,0] = 5
print(f" t is: \n{t}")
print(f" ttr is: \n{ttr}")
print(f"The transposed version was updated, as expected")
print("-"*20)

t[:,:] = Tensor([[10,20]])
print(f" t is: \n{t}")
print(f" ttr is: \n{ttr}")
print(f"The transposed version was updated, as expected")
print("-"*20)

t.add_(Tensor([[2,2]]))
print(f" t is: \n{t}")
print(f" ttr is: \n{ttr}")
print(f"The transposed version was updated, as expected")
print("-"*20)

t.set_(Tensor([[-1,-2]]))
print(f" t is: \n{t}")
print(f" ttr is: \n{ttr}")
print(f"Note that the transposed version was not updated!")