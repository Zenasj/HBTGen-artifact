torch.cumsum(torch.arange(0,6, device='mps'), 0)
# tensor([0, 1, 2, 3, 4, 5], device='mps:0')

torch.cumsum(torch.arange(0,6), 0)
# tensor([ 0,  1,  3,  6, 10, 15])

torch.arange(0,6, device='mps').cumsum(0)
# tensor([0, 1, 2, 3, 4, 5], device='mps:0')

torch.arange(0,6).cumsum(0)
# tensor([ 0,  1,  3,  6, 10, 15])

# monkey-patch cumsum to fallback to CPU
# https://github.com/pytorch/pytorch/issues/89784
import torch
from torch import cumsum, Tensor
torch.cumsum = lambda input, *args, **kwargs: (
  cumsum(input.cpu() if input.device.type == 'mps' else input, *args, **kwargs).to(input.device)
)
orig_cumsum = torch.Tensor.cumsum
def patched_cumsum(self: Tensor, *args, **kwargs):
    return orig_cumsum(self.cpu() if self.device.type == 'mps' else self, *args, **kwargs).to(self.device)
torch.Tensor.cumsum = patched_cumsum

reassuring_message = "monkey-patched cumsum to fallback to CPU, for compatibility on MPS backend; see https://github.com/pytorch/pytorch/issues/89784"

from .cumsum_mps_fix import reassuring_message
print(reassuring_message) # avoid "unused" import :P