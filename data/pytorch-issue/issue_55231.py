import torch

_floating_types = _dispatch_dtypes((torch.float32, torch.float64))
_integral_types = _dispatch_dtypes((torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))
...
_all_types = _floating_types + _integral_types
def all_types():
    return _all_types