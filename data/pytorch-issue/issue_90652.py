import torch
from torch.utils._mode_utils import no_dispatch

with torch._subclasses.FakeTensorMode():
    a = torch.rand([100])
    with torch._subclasses.CrossRefFakeMode():
        # below logic from run_fallback_kernel
        with no_dispatch():
            b = torch.zeros_like(a) # <- exception thrown here