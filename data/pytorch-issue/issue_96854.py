py
import torch
mm = torch.tensor([[[-0.2785],[ 0.0000],[ 0.2785]]])
select_4 = torch.tensor([0, 1, 1, 2, 0, 1, 2])
torch.vmap(torch.index_select, (0, None, None))(mm, -2, select_4)
# Segmentation fault