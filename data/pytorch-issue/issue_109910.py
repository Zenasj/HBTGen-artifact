Python
import torch

class dtype_test(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        dtype=torch.dtype,
    ):
        orig_precision = tensor.dtype
        ctx.orig_precision = orig_precision
        return tensor.to(dtype), dtype
    @staticmethod
    def backward(ctx, dOut):
        return dOut.to(ctx.orig_precision), None, None,

def main():
    x = torch.randn(16, 16, device="cpu", dtype=torch.float32)
    out = dtype_test.apply(x, torch.bfloat16)
    def test_func(x, dtype):
        return dtype_test.apply(x, dtype)
    compiled_func = torch.compile(test_func, fullgraph=True)
    y = compiled_func(x, torch.bfloat16)

if __name__ == "__main__":
    main()