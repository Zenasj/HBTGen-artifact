import torch
from torch import nn
from typing import TypeVar, Type, Any, Optional, Union

T = TypeVar("T", bound="TensorSubclass")

class TensorSubclass(torch.Tensor):
    def __new__(
        cls: Type[T],
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
    ) -> T:
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        return tensor.as_subclass(cls)  # type: ignore[arg-type]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3)  # Example layer with input channel 3

    def forward(self, x: TensorSubclass) -> TensorSubclass:
        return self.conv(x)  # type: ignore

def my_model_function():
    return MyModel()

def GetInput():
    # torch.rand(B, 3, 224, 224, dtype=torch.float32) ← Inferred input shape
    return TensorSubclass(torch.rand(2, 3, 224, 224, dtype=torch.float32))

