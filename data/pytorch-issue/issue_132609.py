import torch


class CustomOp(torch.autograd.Function):
	@staticmethod
	def forward(ctx, i):
		result = i + i
		ctx.save_for_backward = i
		return result

	@staticmethod
	def backward(ctx, grad_output):
		input = ctx.save_for_backward
		detached_inputs = input.detach()
		with torch.enable_grad():
			outputs = detached_inputs * detached_inputs
		output_tensors = []
		grad_tensors = []
		if outputs.requires_grad:
			output_tensors.append(outputs)
			grad_tensors.append(grad_output)
		
		torch.autograd.backward(output_tensors, grad_tensors)

		return detached_inputs.grad()


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