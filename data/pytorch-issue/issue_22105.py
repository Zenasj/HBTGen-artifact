import numpy as np
import torch
import pandas as pd


idx = pd.core.indexes.numeric.Int64Index(np.arange(10000))
arr = torch.zeros(len(idx))

while True:
    arr[idx]