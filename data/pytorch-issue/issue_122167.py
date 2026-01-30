import torch

py
class MySin(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x.sin()

    @staticmethod
    def setup_context(*args, **kwargs):
        pass

    @staticmethod
    def backward(ctx, grad):
        if grad.stride(0) > 1:
            return grad.sin()
        return grad.cos()