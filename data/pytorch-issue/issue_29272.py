import torch

t1 = torch.tensor([[1,2],[3,4]])
t2 = torch.tensor([0.5, 0.5])
print(t1+t2) # tensor([[1.5, 2.5], [3.5, 4.5]]) correct
print(t1[:,0]) # tensor([1, 3]) correct
print(t1[:,0] + t2) # tensor([1.5000, 0.5000]) whaaaat?