import torch
import numpy as np
import random

@torch.compile()
def foo(A):
    # Change me to random.choice instead of np.random.choice to work!
    kwargs = {"UPLO": np.random.choice(["L", "U"])}
    return torch.linalg.eigh(A, **kwargs)

foo(
    torch.tensor(np.random.randint(1, 100, size=(8, 8)),
    dtype=torch.float32)
)