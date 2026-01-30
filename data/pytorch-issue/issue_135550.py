import torch
t = torch.randn(2).to('xpu')
t.data_ptr()
# one possible output is:
-23453392437248