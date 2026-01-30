import torch
                            # ↓↓↓↓↓↓↓↓↓
torch.arange(end=torch.tensor([0, 1, 2])) # Error

import torch
                              # ↓↓↓↓↓↓↓↓↓
torch.arange(start=torch.tensor([0, 1, 2]), end=5) # Error

import torch
                                             # ↓↓↓↓↓↓↓↓↓
torch.arange(start=0, end=5, step=torch.tensor([0, 1, 2])) # Error

import torch
                            # ↓
torch.arange(end=torch.tensor(2)) # tensor([0, 1])
                              # ↓
torch.arange(start=torch.tensor(2), end=5) # tensor([2, 3, 4])
                                             # ↓
torch.arange(start=0, end=5, step=torch.tensor(2)) # tensor([0, 2, 4])

import torch
                            # ↓↓
torch.arange(end=torch.tensor(2.)) # tensor([0., 1.])
                            # ↓↓↓↓↓↓
torch.arange(end=torch.tensor(2.+0.j)) # tensor([0., 1.])
                            # ↓↓↓↓
torch.arange(end=torch.tensor(True)) # tensor([0])

                              # ↓↓
torch.arange(start=torch.tensor(2.), end=5.) # tensor([2., 3., 4.])
                              # ↓↓↓↓↓↓
torch.arange(start=torch.tensor(2.+0.j), end=5.+0.j) # tensor([2., 3., 4.])
                              # ↓↓↓↓↓
torch.arange(start=torch.tensor(False), end=True) # tensor([0])

                                               # ↓↓
torch.arange(start=0., end=5., step=torch.tensor(2.)) # tensor([0., 2., 4.])
                                                       # ↓↓↓↓↓↓
torch.arange(start=0.+0.j, end=5.+0.j, step=torch.tensor(2.+0.j)) # tensor([0., 2., 4.])
                                                    # ↓↓↓↓
torch.arange(start=False, end=True, step=torch.tensor(True)) # tensor([0])

import torch

torch.__version__ # 2.3.0+cu121