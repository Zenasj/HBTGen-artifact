import torch
eq = 'abcdefghijklmnopqrstuvwx,egmpfybhdzAiqcBjaCrntoswxuDv->acijkluvwyzABCD'
x = torch.rand([2] * len('abcdefghijklmnopqrstuvwx'), device='cuda')
y = torch.rand([2] * len('egmpfybhdzAiqcBjaCrntoswxuDv'), device='cuda')
z = torch.einsum(eq, x, y)
# RuntimeError: tensor has too many (>25) dims

x = torch.rand([2] * 28, device='cuda')
y = torch.rand([2] * 28, device='cuda')
# these all work fine:
z = x + y
z = x * y
z = torch.permute(x, tuple(range(27, -1, -1)))
z = torch.cos(x)
z = torch.tensordot(x, y, 14)

# contract every other dimension
z = torch.tensordot(x, y, [tuple(range(0, 28, 2)), tuple(range(0, 28, 2))])