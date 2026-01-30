Variable._execution_engine.run_backward(
        tensors, grad_tensors, retain_graph, create_graph,
        allow_unreachable=True)  # allow_unreachable flag

import torch

class fn_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output.device)
        assert False
        return torch.zeros_like(grad_output)
fn = fn_.apply
a = torch.tensor(1., requires_grad=True, device="cuda")
fn(a).backward()