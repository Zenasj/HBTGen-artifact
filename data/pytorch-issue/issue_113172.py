import torch

def fn(x, y, z, device): 
  x = torch.atan(out=x, input=y)      
  x = torch.t(input=x, )        
  x = torch._C._special.special_bessel_y0(out=x, input=z)# when commented out special_bessel_yo, no error happened.        
  return x

device = 'cpu'
#device = 'cuda'# also encountered this issue!
x = torch.rand([10,9],dtype=torch.float32).to(device)
y = torch.randint(-9223372036854775808, 9223372036854775807, [10, 9], dtype=torch.int64).to(device)
z = torch.rand(9,10) > 0.5
z = z.to(device) 
eag = fn(x, y, z, device)# res on eager mode
opt = torch.compile(fn, mode='default')(x, y, z, device) # res with default torch.comile

same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')