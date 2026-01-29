# torch.rand(B, C, dtype=torch.float16)  # B=1024, C=1024

import torch
from torch import nn

class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, W: torch.Tensor, B: torch.Tensor):
        print(f"fw")
        ctx.wcsplt = torch.ops.aten._cslt_compress(W)
        ctx.wcspltT = torch.ops.aten._cslt_compress(W.t().contiguous())
        ctx.save_for_backward(x)
        return torch.ops.aten._cslt_sparse_mm(ctx.wcsplt, dense_B=x.t(), bias=B, transpose_result=True)
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        print("bw")
        x = ctx.saved_tensors[0]
        return (
            # dx
            torch.ops.aten._cslt_sparse_mm(ctx.wcsplt, grad_out.t(), transpose_result=True),
            # dw
            x.t() @ grad_out,
            # db
            None
        )

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ft = 1024  # From issue's repro code
        self.W = nn.Parameter(torch.randn(ft, ft, device="cuda", dtype=torch.float16))
        self.B = nn.Parameter(torch.randn(ft, device="cuda", dtype=torch.float16))
    
    def forward(self, x):
        return Linear.apply(x, self.W, self.B)

def my_model_function():
    return MyModel()

def GetInput():
    b, ft = 1024, 1024
    return torch.rand(b, ft, device="cuda", dtype=torch.float16)

