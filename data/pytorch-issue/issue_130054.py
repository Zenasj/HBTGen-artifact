import torch

t1 = torch.tensor([[]])
t2 = torch.tensor([])

torch.mv(input=t1, vec=t2)
# tensor([0.])

# Additional
t1 = torch.tensor([[0.]])
t2 = torch.tensor([0.])

torch.mv(input=t1, vec=t2)
# tensor([0.])

import torch

torch.__version__ # 2.3.0+cu121