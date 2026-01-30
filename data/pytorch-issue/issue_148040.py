import torch

@torch.compile(backend="aot_eager")
def fn(static, dynamic):
    return torch.matmul(static, dynamic)  # inner dims coerced by matmul

static = torch.randn(10, 10)
dynamic = torch.randn(10, 10)
torch._dynamo.mark_dynamic(dynamic, 0)
fn(static, dynamic)