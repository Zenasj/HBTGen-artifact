import torch

@torch.library.custom_op(
    "aoti_custom_ops::fn_ret_list_of_single_tensor", mutates_args={}
)
def fn_ret_list_of_single_tensor(x: torch.Tensor) -> list[torch.Tensor]:
    s = x.sum().to(torch.int64)
    return [torch.randn(s.item())]


@fn_ret_list_of_single_tensor.register_fake
def _(x):
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.new_dynamic_size()
    return [torch.randn(i0)]