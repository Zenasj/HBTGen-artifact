import torch
import numpy as np
x_array = np.array([],dtype=np.float16)
x_tensor = torch.from_numpy(x_array)
torch._foreach_mul_((x_tensor,),1.0)