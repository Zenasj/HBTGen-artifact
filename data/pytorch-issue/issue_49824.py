import torch

a = torch.rand(10, requires_grad=True).clone()

# View for which inplace are invalid
b = a.unbind(0)
# Chain of view where the last one is valid for inplace
c = b[0].view_as(b[0])

# This one should fail but does not
c.mul_(2)
# This one properly fails
b.mul_(2)