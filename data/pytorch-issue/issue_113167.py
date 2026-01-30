import torch

def fn(x, device):
    return torch.var_mean(input=x, correction=4.9,dim=-1,keepdim=True)        

device = 'cuda'# tried to set device = 'cpu', and it also get a diff value
x = torch.rand([], dtype=torch.float32).to(device) # as input
eag = fn(x, device) # outputs are tuples
opt = torch.compile(fn, mode='default')(x, device)# when mode = 'max-autotune-no-cudagraphs', this issue also be encountered 

eag = eag[0] # in the puporse of using torch.allclose
opt = opt[0]
same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
    raise ValueError('diff value')