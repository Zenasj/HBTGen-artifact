import torch.nn as nn

import torch

class CumSum(torch.nn.Module):
    def __init__(self):
        super(CumSum, self).__init__()

    def forward(self, x):
        return torch.cumsum(x, dim=0)
    

def main():
    # illegal memory access for fp32
    inp_arr = torch.rand((1000000,)).cuda()
    # no issue for fp32
    # inp_arr = torch.rand((100000,)).cuda()
    
    # illegal memory access for fp64
    # inp_arr = torch.rand((10000,)).cuda().double()
    # no issue for fp64
    # inp_arr = torch.rand((1000,)).cuda().double()

    # illegal memory access, numel=1e6
    # inp_arr = torch.rand((100000, 10)).cuda()
    # illegal memory access, numel=1e6
    # inp_arr = torch.rand((1000, 1000)).cuda()
    # no issue, numel=1e6
    # inp_arr = torch.rand((100, 10000)).cuda()
    # no issue, numel=1e5
    # inp_arr = torch.rand((1000, 100)).cuda()

    cumsum_eager = CumSum()

    with torch.no_grad():
        so_path = torch._export.aot_compile(cumsum_eager, (inp_arr,))
    cumsum_aot_compiled = torch._export.aot_load(so_path, device="cuda")

    cumsum_jit_compiled = torch.compile(cumsum_eager)
    
    with torch.no_grad():
        for _ in range(100):
            out_eager = cumsum_eager(inp_arr)
            out_aot = cumsum_aot_compiled(inp_arr)
            out_jit = cumsum_jit_compiled(inp_arr)

            print(f"eager matches aot: {torch.allclose(out_eager, out_aot)}, max diff: {torch.abs(out_eager - out_aot).max()}")
            print(f"eager matches jit: {torch.allclose(out_eager, out_jit)}, max diff: {torch.abs(out_eager - out_jit).max()}")


if __name__ == "__main__":
    main()