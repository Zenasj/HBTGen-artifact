import torch
import triton

device = "cuda"
input = torch.randn([100], dtype=torch.float, device=device, requires_grad=True)
indices = torch.randint(low=0, high=1, size=(74544,), dtype=torch.int64, device=device)

def gather():
    output = torch.gather(input, dim=0, index=indices)
    output.sum().backward()

triton.testing.do_bench(lambda: gather())