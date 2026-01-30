import torch
import numpy as np

def f(x: torch.Tensor) -> np.ndarray:
	a = x.numpy()
	return a.T

def f(x):
    ___tmp_0 = __compiled_fn_0(x)
    if isinstance(___tmp_0, torch_np.ndarray):
        return ___tmp_0.tensor.numpy()
    else:
        return ___tmp_0