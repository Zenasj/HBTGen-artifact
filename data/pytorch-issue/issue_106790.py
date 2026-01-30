import torch

def clamp_tensor(self, min=None, max=None, *, out):
    print("custom clamp")
    if self.dtype is torch.bool:
        raise Exception()
    self.copy_(out)
    if min is not None:
        out = torch.maximum(out, min)
    if max is not None:
        out = torch.minimum(out, max)
    return out

_aten_lib = torch.library.Library("aten", "IMPL")
_aten_lib.impl("clamp.Tensor_out", clamp_tensor, "CPU")

out = torch.empty((), dtype=torch.bool)
print(torch.clamp(torch.tensor(True), min=torch.tensor(False), max=torch.tensor(False), out=out))