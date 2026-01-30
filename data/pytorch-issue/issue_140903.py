import torch
from collections import namedtuple

CustomDtype = namedtuple("CustomDtype", ["dtype", "higher_dtype"])

class CustomTensor(torch.Tensor):

    _data: torch.Tensor
    custom_dtype: CustomDtype
    __torch_function__ = torch._C._disabled_torch_function_impl
    __slots__ = [
        "_data",
        "custom_dtype",
    ]

    def __new__(
        cls,
        data: torch.Tensor,
        custom_dtype: CustomDtype,
    ):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=custom_dtype.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self.custom_dtype = custom_dtype
        return self

    def __tensor_flatten__(self):
        meta = {
            "custom_dtype": self.custom_dtype,
        }
        return ["_data"], meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors: dict, metadata, outer_size, outer_stride):
        return CustomTensor(
            inner_tensors["_data"],
            metadata["custom_dtype"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
        return func(*args, **kwargs)

def maybe_cast_up(tensor):
    return CustomTensor(tensor._data.to(tensor.custom_dtype.higher_dtype), tensor.custom_dtype)

def maybe_cast_down(tensor):
    return CustomTensor(tensor._data.to(tensor.custom_dtype.dtype), tensor.custom_dtype)

def create_custom_tensor(tensor):
    return CustomTensor(tensor._data, tensor.custom_dtype)

@torch.compile
def create_custom_tensor_cast(tensor):
    tensor = maybe_cast_up(tensor)
    ret = create_custom_tensor(tensor)
    return maybe_cast_down(ret)

print(torch._dynamo.explain(create_custom_tensor_cast)(CustomTensor(torch.randn(1000, dtype=torch.float16), CustomDtype(torch.float16, torch.float32))))

import torch

class CustomTensor(torch.Tensor):

    _data: torch.Tensor
    custom_dtype: tuple
    __torch_function__ = torch._C._disabled_torch_function_impl
    __slots__ = [
        "_data",
        "custom_dtype",
    ]

    def __new__(
        cls,
        data: torch.Tensor,
        custom_dtype: tuple,
    ):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=custom_dtype[0],
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self.custom_dtype = custom_dtype
        return self

    def __tensor_flatten__(self):
        meta = {
            "custom_dtype": self.custom_dtype,
        }
        return ["_data"], meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors: dict, metadata, outer_size, outer_stride):
        return CustomTensor(
            inner_tensors["_data"],
            metadata["custom_dtype"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
        return func(*args, **kwargs)

def maybe_cast_up(tensor):
    return CustomTensor(tensor._data.to(tensor.custom_dtype[1]), tensor.custom_dtype)

def maybe_cast_down(tensor):
    return CustomTensor(tensor._data.to(tensor.custom_dtype[0]), tensor.custom_dtype)

def create_custom_tensor(tensor):
    return CustomTensor(tensor._data, tensor.custom_dtype)

@torch.compile
def create_custom_tensor_cast(tensor):
    tensor = maybe_cast_up(tensor)
    ret = create_custom_tensor(tensor)
    return maybe_cast_down(ret)

print(torch._dynamo.explain(create_custom_tensor_cast)(CustomTensor(torch.randn(1000, dtype=torch.float16), (torch.float16, torch.float32))))