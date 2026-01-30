import torch.nn as nn

import torch
a = torch.nn.Parameter(torch.complex(torch.rand(3), torch.rand(3)))
b = torch.tensor(1.0)
c = a * b
c.real = c.real.clamp(0, 0.1)
c.abs().mean().backward()

import torch
a = torch.nn.Parameter(torch.complex(torch.rand(3), torch.rand(3)))
b = torch.tensor(1.0)
c = a * b
c.real = c.real.clamp_(0, 0.1)
c.abs().mean().backward()

a = torch.complex(torch.rand(1, requires_grad=True), torch.rand(1, requires_grad=True))
a.real = a.real.clamp(0, 0.1)
a.real.backward()

import torch
func_cls=torch.Tensor.clamp
a = torch.nn.Parameter(torch.complex(torch.rand(3), torch.rand(3)))
b = torch.tensor(1.0)
c = a * b
x=c.real
def test():
	tmp_result= func_cls(x,0, 0.1)
	return tmp_result
c.real = test()
c.abs().mean().backward()