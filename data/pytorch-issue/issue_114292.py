import torch

class RegisterPostBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        param_group,
        *inputs,
    ):
        # All tensors in `inputs` should require gradient
        ctx.param_group = param_group
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        ctx.param_group._post_backward()
        return (None,) + grads

hook = functools.partial(_post_backward, param_group)
class RegisterPostBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hook,
        *inputs,
    ):
        # All tensors in `inputs` should require gradient
        ctx.hook = hook
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        ctx.hook()
        return (None,) + grads