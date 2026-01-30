import torch

my_tensor = torch.tensor(7.+5.j)

torch.aminmax(input=my_tensor) # RuntimeError: "aminmax_cpu" not implemented for 'ComplexFloat'

import torch

my_tensor = torch.tensor(7.+5.j)

torch.aminmax(input=my_tensor, dim=0)
torch.aminmax(input=my_tensor, dim=-1)
# torch.return_types.aminmax(
# min=tensor(7.+5.j),
# max=tensor(7.+5.j))

import torch

torch.__version__ # 2.2.1+cu121