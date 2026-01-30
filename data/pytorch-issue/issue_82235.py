import torch.nn as nn
import numpy as np

import torch

for device in ["cuda", "cpu"]:
    inp = torch.rand([1, 9, 2, 2], device=device)
    output = torch.nn.functional.pixel_shuffle(inp, upscale_factor=1)
    print(inp.storage()._cdata == output.storage()._cdata)

# True
# False