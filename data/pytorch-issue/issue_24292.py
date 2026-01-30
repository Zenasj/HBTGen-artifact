import torch

In [2]: print(torch.cuda._initialized)                                                                                                                                                                                                                                                     
False

In [3]: x = torch.randn(10).cuda()                                                                                                                                                                                                                                                         

In [4]: print(torch.cuda._initialized)                                                                                                                                                                                                                                                     
True