import torch.nn as nn

my_tensor = torch.tensor([2., 7., 4.])
                                                     # ↓↓
linear = nn.Linear(in_features=3, out_features=5, bias=10)
                                                     # ↓↓↓
linear = nn.Linear(in_features=3, out_features=5, bias=10.)
                                                     # ↓↓↓↓↓↓↓
linear = nn.Linear(in_features=3, out_features=5, bias=10.+0.j)
linear(input=my_tensor)
# tensor([ 5.1104,  1.2912, -0.8506, -1.9801, -5.2040], grad_fn=<ViewBackward0>)

import torch

torch.__version__ # 2.4.0+cu121