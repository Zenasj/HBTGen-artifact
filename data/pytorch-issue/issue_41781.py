import torch
print (torch.max(torch.zeros([1, 2001]).to(torch.device('cpu')), 1)[1])
print (torch.max(torch.zeros([1, 2001]).to(torch.device('cuda:0')), 1)[1])

tensor([2000])
tensor([0], device='cuda:0')