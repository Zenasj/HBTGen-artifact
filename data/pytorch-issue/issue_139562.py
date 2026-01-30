import torch

tensor1 = torch.tensor([0., 1., 2.])
tensor2 = torch.tensor([0., 1., 2.])

                                          # ↓↓↓↓↓↓↓↓↓↓↓
torch.isclose(input=tensor1, other=tensor2, rtol=4.+0.j) # Error

                                          # ↓↓↓↓↓↓↓↓↓↓↓
torch.isclose(input=tensor1, other=tensor2, atol=3.+0.j) # Error

import torch

tensor1 = torch.tensor([0., 1., 2.])
tensor2 = torch.tensor([0., 1., 2.])

                                          # ↓↓↓↓↓↓  ↓↓↓↓↓↓
torch.isclose(input=tensor1, other=tensor2, rtol=3, atol=4)
# tensor([True, True, True])
                                          # ↓↓↓↓↓↓↓↓↓  ↓↓↓↓↓↓↓↓↓↓
torch.isclose(input=tensor1, other=tensor2, rtol=False, atol=False)
# tensor([True, True, True])

import torch

torch.__version__ # '2.5.1'