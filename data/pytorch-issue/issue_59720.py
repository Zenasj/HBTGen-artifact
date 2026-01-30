import torch
a = torch.load('symeigbug.pt')
print(a.symeig(eigenvectors = True)[1].isnan().any())
# tensor(True)