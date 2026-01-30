import torch

device = torch.device("cuda:0")
x = torch.randn(10, dtype=torch.float32, device=device)
y = torch.randn(10, dtype=torch.float32, device=device)
z = torch.zeros(10, dtype=torch.float32, device=device)

with torch.cuda.device('cuda:1'): # Wrong device
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        z = x + y

    for i in range(3):
        x.normal_()
        y.normal_()
        g.replay()
        print(z) # One would expect it to print different values each iteration, 
                 # but it does not because the current_device is 0 
                 # while all the tensors are on device 1

    print(f'Test passed')