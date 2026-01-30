import torch.nn as nn

import torch
from torch.autograd import Variable

class UnfoldTestModule(torch.nn.Module):
	def __init__(self):
		super(UnfoldTestModule, self).__init__()
		self.unfold = torch.nn.Unfold(kernel_size=(3,3))

	def forward(self, input):
		output = self.unfold(input)
		return output


if __name__ == '__main__':
	my_module = UnfoldTestModule()
	example_input = torch.randn(2, 5, 3, 4)

	traced_script_module = torch.jit.trace(my_module, example_input)
	traced_script_module.save("model.pt")