import torch

def check_fmod(n, dtype = torch.float32, device = 'cpu'): 
    a = torch.randn(n, device = device, dtype = dtype)
    b = torch.randn(n, device = device, dtype = dtype)

    random_index = torch.randint(0, n, size = (1,)).item()
    a[random_index] = torch.tensor(8.0, device = device, dtype = dtype)
    b[random_index] = torch.tensor(2.0e-38, device = device, dtype = dtype) 

    res1 = torch.fmod(a, b)
    print(n, res1[random_index])

for n in range(1, 64):
    check_fmod(n, dtype = torch.float32, device = 'cpu')