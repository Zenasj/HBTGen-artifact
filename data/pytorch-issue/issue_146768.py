import torch.nn as nn

import torch
import os
os.environ["MTL_CAPTURE_ENABLED"]="1"
inp = torch.rand(size=(6, 3, 10, 20), device="mps", dtype=torch.float32)
with torch.mps.profiler.metal_capture("bilinear2d"):
    out = torch.nn.functional.interpolate(x, scale_factor=(1.7,0.9), mode="bilinear")