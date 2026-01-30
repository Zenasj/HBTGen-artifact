import torch

py
In [31]: torch.empty(3, device='mps:0').bernoulli_(0.3)
Out[31]: tensor([1., 1., 0.], device='mps:0')

In [32]: torch.empty(3, device='mps:0').bernoulli_(0.3)
Out[32]: tensor([0., 0., 0.], device='mps:0')

In [33]: torch.empty(3, device='mps:0').bernoulli_(0.3)
Out[33]: tensor([0., 0., 1.], device='mps:0')

In [34]: torch.rand(3, device='mps:0')
Out[34]: tensor([0.3318, 0.2188, 0.4544], device='mps:0')

In [35]: torch.rand(3, device='mps:0')
Out[35]: tensor([0.9184, 0.0832, 0.8531], device='mps:0')

In [36]: torch.randn(3, device='mps:0')
Out[36]: tensor([-1.2882, -0.0142, -0.4158], device='mps:0')

In [37]: torch.randn(3, device='mps:0')
Out[37]: tensor([-1.2882, -0.0142, -0.4158], device='mps:0')

In [38]: torch.randn(3, device='mps:0')
Out[38]: tensor([-1.2882, -0.0142, -0.4158], device='mps:0')

In [39]: torch.__version__
Out[39]: '1.13.0.dev20220523'