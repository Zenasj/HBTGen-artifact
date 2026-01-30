import torch
from typing import TypedDict

class EmptyDict(TypedDict):
    pass

@torch.compile(fullgraph=True)
def fun(x):
    ddd = EmptyDict()
    return x+x

fun(torch.ones(1000))

class LlavaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size, num_channels, height, width)`"""