try:
    import torch.fft   # for python 1.7
except Exception:
    pass

def function_that_needs_new_torch_version():
    ...  # some code that uses torch.fft.fft

import torch
def function_that_still_uses_old_fft():
    torch.fft(...)

function_that_needs_new_torch_version()