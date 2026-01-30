import torch.nn as nn

a = torch.tensor([-1., 1.], requires_grad=True)
b = torch.nn.functional.leaky_relu_(a.clone(), -2)
b.backward(torch.ones(2))
a.grad

import torch
torch.set_anomaly_enabled(True)

a = torch.tensor([-1., 1.], requires_grad=True)
b = torch.nn.functional.leaky_relu_(a.clone(), -2)
b.backward(torch.ones(2))
a.grad

# output:
# tensor([1., 1.])