import torch

@register_meta(aten._local_scalar_dense)
def meta_local_scalar_dense(self:Tensor):
    torch._check(
        not self.is_meta,
        lambda: "aten::_local_scalar_dense operator cannot be called on meta tensors.",
    )
    return torch.empty_like(self)