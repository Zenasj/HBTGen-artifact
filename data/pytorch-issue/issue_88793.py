import torch
input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
storage = torch.CharStorage(input_data)
torch.save(input_data, storage)

import torch
import numpy as np
x = torch.randint(10, (3, 3), dtype=torch.float)
torch.save(x.cpu().numpy(), x.cpu().numpy())