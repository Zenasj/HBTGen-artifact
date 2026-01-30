import torch.nn as nn

import torch

torch.manual_seed(0)

m = torch.nn.Linear(1, 2, bias = False)

x = torch.tensor([[0.0]])
y = torch.tensor([[float('nan'), 0.0]])
y_mask = ~torch.isnan(y)
y_pred = m(x)

def loss_fn(y_pred, y):
	# problem shows up when using l2 or log loss (but not l1)
	return torch.square(y_pred - y)

# these two branches should be equivalent; they aren't
if True: # (1) mask after loss_fn
	# doing it this way results in NaNs in the grad
	l = loss_fn(y_pred, y)[y_mask]
else:    # (2) mask before loss_fn
	l = loss_fn(y_pred[y_mask], y[y_mask])

# loss is fine: no NaNs
print("loss:", l)
assert not torch.any(torch.isnan(l))

l.sum().backward()

# grad should not have NaNs
assert not torch.any(torch.isnan(m.weight.grad))