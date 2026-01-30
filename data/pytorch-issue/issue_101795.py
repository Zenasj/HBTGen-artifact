py
from time import time
import numpy as np
import matplotlib.pyplot as plt

import torch

torch_array = torch.randn(1000, 150)

def plot_hist(array):
    init = time()
    plt.figure()
    plt.hist(array)
    print(f"Time to plot: {time() - init:.2f} s")
    plt.show()

plot_hist(torch_array.ravel())  # Takes around 2 seconds
plot_hist(np.array(torch_array.ravel()))  # Takes around 0.04 seconds