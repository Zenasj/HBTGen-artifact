import torch

data = torch.load('E15_s1_08.pt')

torch.save(data.type(torch.uint8),'E15_s1_08_test.pt') 

data = torch.load('E15_s1_08_test.pt')

# `data` is the tensor
torch.save(data.type(torch.float32),'E15_s1_08.pt')