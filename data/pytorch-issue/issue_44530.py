py
import torch
from torch.distributions import Normal
from torch.distributions.transforms import AffineTransform, CatTransform

prior = Normal(torch.zeros(16, 8), torch.ones(16, 8))
t1 = AffineTransform(0, 1, event_dim=1)
t2 = AffineTransform(0, 1, event_dim=1)

tc = CatTransform([t1, t2], dim=1, lengths=[4, 4])

ps = prior.sample()
tps = tc(ps)  # Works fine, returns a tensor of size ([16, 8])
lad = tc.log_abs_det_jacobian(ps, tps)  # This line throws an error:

py
t = CatTransform([ExpTransform(), identity_transform], lengths=[3,3], dim=-1)
x = torch.rand(B, 3+3)
assert t(x).shape == (B, 3+3)

py
t = CatTransform([ExpTransform(), identity_transform], lengths=[B,B], dim=-2)
x = torch.rand(B+B, 3)
assert t(x).shape == (B+B, 3)