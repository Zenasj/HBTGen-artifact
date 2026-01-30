import numpy as np
import torch

t = torch.randn(18, 7)
dist_matrix = torch.cdist(t, t) # <-- Segmentation fault here

import numpy as np
import torch

t = torch.randn(30, 7)
dist_matrix = torch.cdist(t, t) # <-- No segmentation fault

import torch
import numpy as np

t = torch.randn(18, 7)
dist_matrix = torch.cdist(t, t) # <-- No segmentation fault

import numpy as np
import torch

torch.set_num_threads(1)
t = torch.randn(18, 7)
dist_matrix = torch.cdist(t, t) # <-- No segmentation fault, usually