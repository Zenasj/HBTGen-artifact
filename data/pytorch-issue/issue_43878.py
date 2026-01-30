import torch

if __name__ == "__main__":
	# for Categorical
	probs = torch.tensor([[0.4, 0.6],[0.3, 0.7]])
	dist = torch.distributions.Categorical(probs)

	a = torch.tensor([[1],[0]])
	output = dist.log_prob(a)
	print(output.size())  # output torch.Size([2, 2]),expecting torch.Size([2, 1])

	a = torch.tensor([1,0])
	output = dist.log_prob(a)
	print(output.size())# output torch.Size([2]),OK


	# for normal
	means = torch.tensor([[0.0538], [0.0651]])
	stds = torch.tensor([[0.7865], [0.7792]])
	dist = torch.distributions.Normal(means,stds)

	a = torch.tensor([[1.2],[3.4]])
	output = dist.log_prob(a)
	print(output.size())# output torch.Size([2, 1]),OK

	a = torch.tensor([1.2,3.4])
	output = dist.log_prob(a)
	print(output.size())# output torch.Size([2, 2]), expecting torch.Size([2])

py
from pyro.distributions.utils import broadcast_shape

# for Categorical
probs = torch.tensor([[0.4, 0.6],[0.3, 0.7]])
dist = torch.distributions.Categorical(probs)
assert dist.batch_shape == (2,)
assert dist.event_shape == ()

a = torch.tensor([[1],[0]])
assert a.shape == (2, 1)  # this is effectively a batch shape
output = dist.log_prob(a)
assert output.shape == (2, 2)
assert output.shape == broadcast_shape(dist.batch_shape, a.shape)

a = torch.tensor([1,0])
output = dist.log_prob(a)
assert output.shape == (2,)
assert output.shape == broadcast_shape(dist.batch_shape, a.shape)