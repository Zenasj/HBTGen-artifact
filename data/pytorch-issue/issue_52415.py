import torch
from torch.utils.checkpoint import checkpoint

def foo():
    return 1, None, torch.rand(10)

checkpoint(foo)

class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype):
        ctx.reduce_dtype = reduce_dtype

        output_list = list(torch.empty_like(tensor) for _ in range(_CONTEXT_PARALLEL_GROUP_SIZE))
        handle = dist.all_gather(output_list, tensor, _CONTEXT_PARALLEL_GROUP, async_op=True)
        out = [*output_list, handle]
        return tuple(out)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        grad_dtype = grad_list[0].dtype 
        grad_input = torch.empty_like(grad_list[_CONTEXT_PARALLEL_RANK])
        dist.reduce_scatter(grad_input, grad_list, dist.ReduceOp.SUM, _CONTEXT_PARALLEL_GROUP)
        return grad_input.to(grad_dtype), None, None