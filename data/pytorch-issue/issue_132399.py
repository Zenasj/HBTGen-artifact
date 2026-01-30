import torch


class CustomOp(torch.autograd.Function):
	@staticmethod
	def forward(ctx, i):
		result = i + i
		tmp = (result, i)
		ctx.cached_data = tmp
		return result

	@staticmethod
	def backward(ctx, grad_output):
		for t in ctx.cached_data:
			if hasattr(t, 'saved_data'):
				result = grad_output * t
			else:
				result = grad_output * 2
		ctx.cached_data = None
		return result


@torch.compile(backend='inductor', fullgraph=True) 
def custom_function(x):
	x = CustomOp.apply(x)
	return x


def main():
    a = torch.tensor([1., 2.]).requires_grad_(True)
    r = custom_function(a)
    l = torch.sum(r)
    l.backward(torch.tensor(1.))


if __name__ == "__main__":
    main()