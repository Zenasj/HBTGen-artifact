import torch.nn as nn
input_size=[1,1,0] 
m = nn.Linear(in_features=0,out_features=0,bias=True)
m(torch.randn(input_size))


torch.compile(m.to('cuda'))(torch.randn(input_size).to('cuda')) # With torch.compile(), a LoweringExceptionis raised: