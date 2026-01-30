import torch

for d in range(2, 10):
    a = torch.randn(64, d, d)
    b = torch.randn(64, d)
    y1 = torch.einsum("bij,bj->b", a, b)
    y2 = torch.einsum("bij,bj->bi", a, b).sum(dim=1)
    print(f"{d=}, {torch.allclose(y1, y2)=}, {torch.linalg.norm(y1 - y2)=}")

y = torch.bmm(a, b.unsqueeze(-1)).sum(dim=1).squeeze()

y = torch.einsum("bij,bj->bi", a, b).sum(dim=1)