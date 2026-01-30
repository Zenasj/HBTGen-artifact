import torch

torch.manual_seed(123)

for device in [torch.device("cuda:0"), torch.device("cpu")]:

    a = torch.rand(10, 20, 21).to(device)
    ix = torch.arange(10).to(torch.int64).to(device)
    iy = torch.arange(10).to(torch.int64).to(device)
    ib = torch.arange(10).to(torch.int64).to(device)

    ix[0] = 900
    print("Testing device: %s"%(device))
    print(a[ib, ix, iy])