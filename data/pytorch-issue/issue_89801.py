import torch

class FooFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices,
        offsets,
        weights,
    ) -> torch.Tensor:
        with torch.enable_grad():
            return torch.ones(2, requires_grad=True)

    @staticmethod
    def backward(ctx, dout):
        return None, None, [torch.ones(1)]