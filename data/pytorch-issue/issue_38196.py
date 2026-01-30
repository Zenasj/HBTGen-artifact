import torch
import numpy as np
import random

tensors_list = [torch.Tensor(1,3), torch.Tensor(1,3)]
list_to_tensor = torch.Tensor(tensors_list)

np_list = [np.random.rand(1,3), np.random.rand(1,3)]
list_to_tensor = torch.Tensor(np_list)