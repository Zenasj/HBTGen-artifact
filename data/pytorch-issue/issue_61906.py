import torch

torch.testing.assert_close(
    complex(float("nan"), 0), 
    complex(0, float("nan")), 
    equal_nan=True
)