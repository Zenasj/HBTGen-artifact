import torch
torch.use_deterministic_algorithms(True)
a = torch.randn(10, 10, 10).cuda()
b = torch.randn(10, 10, 10).cuda()
c = torch.bmm(a, b)
d = torch.bmm(a.to_sparse(), b)

...
torch.use_deterministic_algorithms(True)
out = torch.bmm(reshaped_x, self.weight1)
torch.use_deterministic_algorithms(False)
...