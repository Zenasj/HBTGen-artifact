import torch
import numpy as np
# 1.24.0
print(np.__version__)

t = torch.randn(2, 3).numpy()
torch.save(t, "numpy.pt")
# ['numpy.core.multiarray._reconstruct', 'numpy.dtype', 'numpy.ndarray']
print(torch.serialization.get_unsafe_globals_in_checkpoint("numpy.pt"))


# This fails despite np.dtype being allowed
# _pickle.UnpicklingError: Weights only load failed. Re-running `torch.load` with `weights_only` set to `False` will likely 
# succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
# Please file an issue with the following so that we can make `weights_only=True` compatible with your use case: 
# WeightsUnpickler error: Can only build Tensor, Parameter, OrderedDict or types allowlisted via `add_safe_globals`,
# but # got <class 'numpy.dtype[float32]'>
with torch.serialization.safe_globals([np.dtype, np.ndarray, np.core.multiarray._reconstruct]):
    torch.load("numpy.pt")

# This succeeds
with torch.serialization.safe_globals([np.dtype, np.ndarray, np.core.multiarray._reconstruct, type(np.dtype(np.float32))]):
    torch.load("numpy.pt")

# This succeeds
with torch.serialization.safe_globals([np.dtype, np.ndarray, np.core.multiarray._reconstruct]):
    torch.load("numpy.pt")