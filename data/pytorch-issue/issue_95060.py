3
import torch
t = torch.tensor([1,2,3,4,5])
print(t.to("cuda:0"))