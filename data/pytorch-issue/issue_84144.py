import torch

x = torch.randn(669, 128, 50259, device="cuda")
print("size", x.nelement() / (1024 ** 3))
torch.softmax(x, dim=-1)[-5:].sum(dim=-1) # returns either 0 or garbage for last several indexes (presumably the zeros also correspond to copied in garbage)