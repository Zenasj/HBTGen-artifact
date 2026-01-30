import torch
import numpy as np

np.mean([torch.tensor(1., device=torch.device("cuda")), torch.tensor(2.4, device=torch.device("cuda"))])

tensor_list = [torch.tensor(1., device=torch.device("cuda")), torch.tensor(2.4, device=torch.device("cuda"))]
np.mean([tensor.cpu() for tensor in tensor_list])