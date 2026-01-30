import torch
a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
q, r = torch.qr(a)
torch.mm(q, r).round()
torch.mm(q.t(), q).round()
a = torch.randn(3, 4, 5)
q, r = torch.qr(a, some=False)
torch.allclose(torch.matmul(q, r), a)
torch.allclose(torch.matmul(q.transpose(-2, -1), q), torch.eye(5))