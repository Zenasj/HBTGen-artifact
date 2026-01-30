import torch

r = torch.randn(10, dtype=torch.float32)                                                                                                                                                                                                                                           
d = torch.randn(10, dtype=torch.float64)                                                                                                                                                                                                                                           
r.add_(d)