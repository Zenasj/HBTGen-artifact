import torch


'''
torch.jit.frontend.NotSupportedError: unsupported binary operator: BitAnd
@torch.jit.script
def fun1(a, b):
	return a & b# RuntimeError: expected a boolean expression for condition but found Tensor, to use a tensor in a boolean expression, explicitly cast it with `bool()`
         ~~~ <--- HERE
'''
@torch.jit.script
def fun1(a, b):
	return a & b


'''
RuntimeError: 
expected a boolean expression for condition but found Tensor, to use a tensor in a boolean expression, explicitly cast it with `bool()`:
@torch.jit.script
def fun2(a, b):
	return a and b
        ~ <--- HERE
'''
@torch.jit.script
def fun2(a, b):
	return a and b


if __name__ == '__main__':
	a = torch.ByteTensor([0])
	b = torch.ByteTensor([1])
	fun1(a, b)
	fun2(a, b)