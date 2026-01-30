import random

py
import torch
from tqdm import tqdm
rows = torch.rand(2_000_000, 2)
_ = [[str(x) for x in row] for row in tqdm(rows)]
#^ Takes 7 seconds to _start_
#^ Averages 2000 it/s

py
import numpy as np
from tqdm import tqdm
rows = np.random.rand(2_000_000, 2)
_ = [[str(x) for x in row] for row in tqdm(rows)]
#^ Starts immediately
#^ Averages 270_000 it/s. Around 130x faster.