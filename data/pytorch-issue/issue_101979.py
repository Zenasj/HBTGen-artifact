import numpy as np

import torch


_result_type_dict = {
    torch.float16: {
        torch.float32: torch.float32,
    },
}

@torch.compile
def f():
    return torch.randn(3, dtype=_result_type_dict[torch.float16][torch.float32])

f()

import torch_np as np
import torch


@torch.compile
def mandelbrot_numpy(max_iter=200):
    # Define the boundaries of the complex plane
    xn = 450
    yn = 375
    xmin = -2.25
    xmax = 0.75
    ymin = -1.25
    ymax = 1.25

    # Create the grid of complex numbers
    x_values = np.linspace(xmin, xmax, xn, dtype=np.float64)
    y_values = np.linspace(ymin, ymax, yn, dtype=np.float64)
    rx, iy = np.meshgrid(x_values, y_values, indexing='xy')

    x = rx.copy()
    y = iy.copy()
    mask = np.zeros_like(x)
    for i in range(max_iter):
        x_prev = x
        y_prev = y
        x = x_prev**2 - y_prev**2 + rx
        y = 2*x_prev*y_prev + iy
        inside = np.sqrt(x**2 + y**2) <= 2
        mask+=inside
    return mask

mandelbrot_numpy()

pykey = key.as_python_constant()
key = pykey if not isinstance(pykey, torch.dtype) else key.source # maybe too specific?
ODictGetItemSource(self.source, key)