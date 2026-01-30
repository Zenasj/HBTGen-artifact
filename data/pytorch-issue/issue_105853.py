import torch

x = torch.tensor([-8.4784-1.7658j])
y = torch.tensor([-8.4784-1.7658j])
ans = torch.compile(torch.matmul)(x, y)
out = torch.empty_like(ans)
torch.compile(torch.matmul)(x, y, out=out)
torch.testing.assert_close(ans, out) # fails

out = torch.compile(torch.matmul)(x, y)
torch.testing.assert_close(ans, out) # success