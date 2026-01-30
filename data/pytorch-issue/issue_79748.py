import numpy as np

import torch
import torch.autograd.forward_ad as fwAD
from torch.testing._internal.composite_compliance import generate_cct

CCT = generate_cct()

for wrap_p, wrap_t in ((False, False), (False, True), (True, False),
                       (True, True)):
    p = torch.randn(1, 1)
    t = torch.randn(1, 1)

    if wrap_p:
        p = CCT(p)
    if wrap_t:
        t = CCT(t)

    try:
        with torch.autograd.forward_ad.dual_level():
            inp = fwAD.make_dual(p, t)
            # inp.squeeze_(0)  # Fails
            inp.unsqueeze_(0)  # Fails
    except Exception as e:
        # Only Fails when both are CCT
        print(wrap_p, wrap_t)
        raise e