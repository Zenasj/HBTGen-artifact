import torch

a = torch.arange(10, dtype = torch.float)

#append a singleton dimension to get shape (10,1)
b = torch.unsqueeze(a, 1)

#correct output: 285
torch.dot(a,a)

#on CPU, throws an error due to 2D tensor
#RuntimeError: 1D tensors expected, but got 1D and 2D tensors
torch.dot(a,b)

#on CPU, throws an error due to non-conforming sizes
#RuntimeError: inconsistent tensor size, expected tensor [5] and src [10] to have the same number of elements, but got 5 and 10 elements respectively
torch.dot(a[0:5], a)

device = torch.device("mps")
a2 = a.to(device)
b2 = b.to(device)

#doesn't throw an error and returns incorrect result 2025
torch.dot(a2,b2)

#expanding b2 to a 10x10 tensor gives the same result
torch.dot(a2, b2.expand((-1, 10)))

#also gives 2025: so maybe torch.dot reduces dimension 0 with a sum,
#broadcasts to the right size (if needed), then takes the dot product
torch.dot(a2, torch.sum(b2))

#transpose: as long as last dimension matches, gives correct result
torch.dot(a2,b2.t())

#both args are 1-D tensors but not compatible sizes
#this crashes the kernel entirely
torch.dot(a2[0:5], a2)