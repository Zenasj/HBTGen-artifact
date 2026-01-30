3
import torch

t_real = torch.tensor([-1.0], requires_grad=True)
l = torch.log(t_real) # This is NaN, as expected.
l.backward()
print(t_real.grad) # This is -1, which is not expected.