import torch
import time

torch.manual_seed(0)
t = torch.randint(500, (10000000, ))
t = torch.sort(t)[0]

start = time.time()
uniques, inverse, counts = torch.unique_consecutive(t, dim=0, return_inverse=True, return_counts=True)
end = time.time()
print("torch.unique_consecutive(dim=0) time:", end - start)

start = time.time()
uniques2, inverse2, counts2 = torch.unique_consecutive(t, return_inverse=True, return_counts=True)
end = time.time()
print("torch.unique_consecutive() time:", end - start)


t = torch.randint(500, (10000000, 2))
t = torch.sort(t)[0]

start = time.time()
uniques, inverse, counts = torch.unique_consecutive(t, dim=0, return_inverse=True, return_counts=True)
end = time.time()
print("torch.unique_consecutive(dim=0) time:", end - start)

start = time.time()
uniques, inverse, counts = torch.unique_consecutive(t, dim=1, return_inverse=True, return_counts=True)
end = time.time()
print("torch.unique_consecutive(dim=1) time:", end - start)