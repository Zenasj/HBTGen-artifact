import torch
import numpy as np

torch.asarray(1., dtype=torch.int32)  # Works
torch.asarray(np.array(1.), dtype=torch.int32)  # Fail
torch.asarray(np.array([1.]), dtype=torch.int32)  # Works

# Works
torch.asarray(1)
torch.asarray(np.array([1]))

# Fail
torch.asarray(np.array(1))
torch.asarray(np.zeros(shape=()))