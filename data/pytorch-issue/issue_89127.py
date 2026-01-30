import torch

mps = torch.device('mps')
cpu = torch.device('cpu')

for i in range(100):
    print(torch.any(torch.isnan(torch.normal(0, 1, size=(1000, 1000), device=mps))))
# occasionally prints True

for i in range(100):
    print(torch.any(torch.isnan(torch.normal(0, 1, size=(1000, 1000), device=cpu).to(mps))))
# doesn't print True