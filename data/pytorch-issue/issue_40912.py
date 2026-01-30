def _bn_function_factory(norm, relu, conv):
	def bn_function(inputs):
		concated_features = inputs
		bottleneck_output = relu(norm(conv(concated_features)))
		return bottleneck_output
	return bn_function

class NET(nn.Module):
	def __init__(self, emb_dims=512):
		super(NET, self).__init__()
		self.k = k
		self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
		self.relu1 = nn.ReLU(inplace=True)
		self.bn1 = nn.BatchNorm2d(64)
	def forward(self, x):
		bn_function = _bn_function_factory(self.bn1, self.relu1, self.conv1)
		batch_size, num_dims, num_points = x.size()
		x = cp.checkpoint(bn_function, x)

model.eval()
src = torch.rand(16, 3, 512).cuda()
tgt = torch.rand(16, 3, 512).cuda()
model_output = torch.jit.trace(model, (src, tgt))
output = model_output(src, tgt)
model_output.save('checkpoints/' + args.exp_name + '/frozen_model.pt')

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


def _bn_function_factory(norm, relu, conv):
	def bn_function(inputs):
		_features = inputs
		_output = relu(norm(conv(_features)))
		return _output

	return bn_function


class NET(nn.Module):
	def __init__(self):
		super(NET, self).__init__()
		self.conv = nn.Conv2d(6, 64, kernel_size=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.bn = nn.BatchNorm2d(64)

	def forward(self, x):
		bn_function = _bn_function_factory(self.bn, self.relu, self.conv)
		x = cp.checkpoint(bn_function, x)
		# x = bn_function(x) # This way works.
		return x


if __name__ == '__main__':
	model = NET()
	model.eval()
	input = torch.rand(16, 6, 128, 20)
	model_output = torch.jit.trace(model, input)
	output = model_output(input)
	model_output.save('frozen_model.pt')