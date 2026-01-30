import torch
b, ft = 1024, 1024
class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, W: torch.Tensor, B: torch.Tensor):  # type: ignore[override]
        print(f"fw")
        ctx.wcsplt = torch.ops.aten._cslt_compress(W)
        ctx.wcspltT = torch.ops.aten._cslt_compress(W.t().contiguous())
        ctx.save_for_backward(x)
        return torch.ops.aten._cslt_sparse_mm(ctx.wcsplt, dense_B=x.t(), bias=B, transpose_result=True)
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
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
x = torch.randn([b, ft], device="cuda", dtype=torch.float16, requires_grad=True)
W = torch.randn([ft, ft], device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.randn([ft], device="cuda", dtype=torch.float16, requires_grad=True)
class Ctx:
    def save_for_backward(self, *x):
        self.saved_tensors = x
print("running bw on same thread")
ctx = Ctx()
out = Linear.forward(ctx, x, W, B)
assert out.shape == (b, ft)
Linear.backward(ctx, out)
print("running bw on different thread")
out = Linear.apply(x, W, B)
out.backward(out)