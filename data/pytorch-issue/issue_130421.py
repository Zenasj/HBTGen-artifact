torch.randint(low=torch.tensor(-5),
              high=torch.tensor([5]),
              size=(torch.tensor([[3]]),
              torch.tensor([[[2]]])))
# tensor([[-2, 3],
#         [-4, 4],
#         [2, 1]])

import torch

torch.__version__ # 2.3.0+cu121