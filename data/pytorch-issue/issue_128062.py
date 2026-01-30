import torch
torch.use_deterministic_algorithms(True)
for i in tqdm(range(2, 1000)):
    a = torch.rand(i+1, i)
    b = torch.rand(i, i)
    if not torch.equal((a @ b)[0:1, :], (a[0:1, :]) @ b):
        print(i)
        break