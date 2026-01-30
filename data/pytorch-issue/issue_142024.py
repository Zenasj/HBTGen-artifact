import torch

@torch.compile(backend="eager", fullgraph=True)
def fn(aot6_sub_58, aot6_mul_170):
    aot6_unsqueeze_14 = torch.ops.aten.unsqueeze.default(aot6_mul_170, 1)
    return torch.ops.aten.mul.Tensor(aot6_sub_58, aot6_unsqueeze_14)

aot6_sub_58 = torch.randn(2, 1)
torch._dynamo.decorators.mark_unbacked(aot6_sub_58, 1)
aot6_mul_170 = torch.randn(2)

fn(aot6_sub_58, aot6_mul_170)