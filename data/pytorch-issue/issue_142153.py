import torch
import numpy as np

torch.save("bla", np_array)
# checkpoint has GLOBAL "np.core.multiarray" "_reconstruct"

with safe_globals([np.core.multiarray._reconstruct]):
    torch.load("bla")