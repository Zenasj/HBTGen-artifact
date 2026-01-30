import torch

torch.manual_seed(0)
a = torch.randn((100, 100), requires_grad=True)
a = a + a.T
torch.lobpcg(a, largest=False)