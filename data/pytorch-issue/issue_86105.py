py
from typing import Type, TypeVar, Any, Optional, Union

import torch

T = TypeVar("T", bound="TensorSubclass")


class TensorSubclass(torch.Tensor):
    def __new__(
        cls: Type[T],
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
    ) -> T:
        return torch.as_tensor(data, dtype=dtype, device=device).as_subclass(cls)