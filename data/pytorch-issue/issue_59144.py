import torch

cauchy = Cauchy(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())

for i in range(400):

  q = cauchy.sample((50000,2))

  try:
    assert torch.isinf(q).any() == False
  except:
    print(q[torch.isinf(q)])
    raise