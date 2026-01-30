import torch

def fn(x, y, z):
    x = torch.zeros_like(x)
    return x.index_put_([y], z, True)
    # return x + 1

x = torch.zeros((512, 512), dtype=torch.bool, device='cuda')
y = torch.arange(512, dtype=torch.int64, device='cuda')
z = torch.ones((512, 512), dtype=torch.bool, device='cuda')

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        fn(x, y, z)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    fn(x, y, z)