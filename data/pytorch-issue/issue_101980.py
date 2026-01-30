import torch


def _result_type_dict(dtype):
    return {bool: torch.float32}[dtype]

@torch.compile
def f():
    return torch.randn(3, dtype=_result_type_dict(bool))

f()