import torch
from torch._dynamo.utils import maybe_enable_compiled_autograd
from torch.cuda.amp import custom_bwd, custom_fwd


class CustomOp(torch.autograd.Function):
	@staticmethod
	@custom_fwd
	def forward(ctx, i):
		result = i @ i
		torch._dynamo.graph_break() 
		return result

	@staticmethod
	@custom_bwd
	def backward(ctx, grad_output):
		return grad_output * 2.0


def custom_function(x):
	x = CustomOp.apply(x)
	return x



def fn(x):
	x = x * x
	y = custom_function(x)
	res = x @ y

	return res


x = torch.randn((1000, 1000), dtype=torch.float32, device="cpu").requires_grad_(True)

with torch.autocast(dtype=torch.bfloat16, device_type="cpu", enabled=True):
	with maybe_enable_compiled_autograd(True):
		fn = torch.compile(fn)
		r = fn(x)
		l = torch.sum(r)
		l.backward()