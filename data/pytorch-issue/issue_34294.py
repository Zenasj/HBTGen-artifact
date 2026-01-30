import torch

torch.cat

torch.stack

__torch_function__

Tensor

torch.cat

__torch_function__

class TensorLike:
    def __torch_function__(func, args=(), kwargs=None):
        print("{} call to __torch_function__ of TensorLike".format(func))

torch.cat

torch.cat((TensorLike(), TensorLike()))

torch.cat((TensorLike(),))

__torch_function__

TensorLike

torch.cat(TensorLike())

torch.cat

torch.cat

_get_overloaded_args

torch/_overrides.py

__torch_function__

torch.cat

test/test_overrides.py

generate_tensor_like_override_tests

torch.cat

TensorLike()

TensorLike

tensors

func_args

[TensorLike()]

[(TensorLike(), TensorLike())]

numpy

__torch_function__

torch

from torch._overrides import has_torch_function, handle_torch_function

original_cat = torch.cat


def cat(tensors, dim=0, out=None) -> Tensor:
    if not torch.jit.is_scripting():
        if any(type(t) is not Tensor for t in tensors) and has_torch_function(tensors):
            return handle_torch_function(cat, tensors, *tensors, dim=dim, out=out)
    return original_cat(tensors, dim=dim, out=out)


torch.cat = cat