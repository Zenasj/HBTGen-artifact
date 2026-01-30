import torch

emb1, emb2, cdist_grad = torch.load('cdist_grad.pt')

emb1.retain_grad()
d = torch.cdist(emb1, emb2)
d.backward(cdist_grad)

print(emb1.grad[0, 17])