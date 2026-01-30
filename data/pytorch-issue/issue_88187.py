import torch
from functorch.dim import dims

t = torch.rand(2, 3, 4)
# this works
assert t.permute(dims=(1, 0, 2)).shape == t.permute(1, 0, 2).shape

d = dims(1)
t_fc = torch.rand(1, 2, 3, 4)[d]
# this doesn't
assert t_fc.permute(dims=(1, 0, 2)).shape == t_fc.permute(1, 0, 2).shape

# this is fine
torch.testing.assert_close(t_fc.permute(dims=(1,0,2))._tensor, t_fc._tensor)

# whereas this raises an AssertionError as expected
torch.testing.assert_close(t_fc.permute(1,0,2)._tensor, t_fc._tensor)