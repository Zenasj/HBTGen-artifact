import torch

cpu = torch.LongTensor([[2,3,0,3,3], [1,3,1,3,1]])
cpu.argmax(dim=1) # tensor([4, 3])
cuda = torch.cuda.LongTensor([[2,3,0,3,3], [1,3,1,3,1]])
cuda.argmax(dim=1) #tensor([4, 1], device='cuda:0')

cpu = torch.LongTensor([[2,3,0,3,0], [1,3,1,3,1]])
cpu.argmax(dim=1) # tensor([4, 1]). 
cuda = torch.cuda.LongTensor([[2,3,0,3,0], [1,3,1,3,1]])
cuda.argmax(dim=1) # tensor([1, 1, device='cuda:0')