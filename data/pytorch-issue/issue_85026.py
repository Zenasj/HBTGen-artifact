import torch
print(torch.__version__)    #  1.13.0.dev20220913+cpu
torch.lu_unpack(LU_pivots=torch.ones((16,0), dtype=torch.int32), LU_data=torch.ones((0,20,4,12)))

import torch
print(torch.__version__)    #  1.13.0.dev20220913+cpu
torch.lu_unpack(LU_pivots=torch.tensor([-10000000], dtype=torch.int32), LU_data=torch.tensor([[1]]))   # segfault