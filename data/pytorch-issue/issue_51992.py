import torch
device = "cuda"
dtype = torch.float32
grads = [torch.ones(3,2, device=device, dtype=dtype), torch.ones(3, device=device, dtype=dtype)]
square_avg = [torch.zeros(3,2, device=device, dtype=dtype), torch.zeros(3, device=device, dtype=dtype)]
alpha=.99
torch._foreach_mul_(square_avg, alpha)
torch._foreach_addcmul_(square_avg, grads, grads, value=1 - alpha) # should be 0.01
#print(square_avg[0], square_avg[1])
avg = torch._foreach_sqrt(square_avg)
print("avg: ", avg[0], avg[1]) #should be 0.1, can be whatever depending in previous prints. With this version it's 0