import torch

def randint_fn(high, size, out):
  return torch.randint(high, size, out=out)

opt_model = torch.compile(randint_fn)

out1 = torch.empty(10)
opt_model(17, (10, ), out1)
print(out1)

out2 = torch.empty(12)
opt_model(17, (12, ), out2)
print(out2)