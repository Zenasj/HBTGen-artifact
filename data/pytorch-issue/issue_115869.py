import torch
t = torch.tensor([0.1, 0.3])
print("cuda")
c = t.cuda() #This line freezes whole process ()
# OR
c = t.to(torch.device('cuda:0'))
# OR
c = t.to('cuda:0')