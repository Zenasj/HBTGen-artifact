import torch
import numpy as np

torch.serialization.add_safe_globals([np.dtype])