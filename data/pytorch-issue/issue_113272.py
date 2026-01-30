import torch

def fn(x, device):
  return torch.empty_like(requires_grad=True, input=x)        


device = 'cpu'
#device = 'cuda'# this can trigger diff value, too!
x = torch.rand([], dtype=torch.float32)

eag = fn(x, device)
opt = torch.compile(fn, mode='max-autotune-no-cudagraphs')(x, device)# no bug if set mode 'default'

same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')