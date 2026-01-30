import torch

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

t = torch.rand(256, 100000)
s1 = torch.get_rng_state()
_ = torch.multinomial(t, 1)
s2 = torch.get_rng_state()

sum(s2-s1)

import torch

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

t = torch.rand(256, 100000).to("cuda")
s1 = torch.cuda.get_rng_state()
_ = torch.multinomial(t, 1)
s2 = torch.cuda.get_rng_state()

sum(s2-s1)