import torch.nn as nn

import torch
import time
import code

import torch._dynamo
torch._dynamo.config.suppress_errors = True

n_iters = 100


def func1() :

  # shapes = [(19, 1024), (18, 1024), (22, 1024), (19, 1024), (13, 1024),
  #           (18, 1024), (21, 1024), (17, 1024), (22, 1024), (19, 1024), (11, 1024)]
  shapes_q = [(torch.randint( low=1, high=1024, size=(1,)).item(),1024) for _ in range(512)]
  shapes_kv = [(torch.randint( low=1, high=1024, size=(1,)).item(),1024) for _ in range(512)]

  al = [torch.randn( *shape, device="cuda", dtype=torch.float16) for shape in shapes_q]
  a = torch.nested.as_nested_tensor( al, layout=torch.jagged)

  bb = torch.tensor( shapes_q)
  print( f'XX : {bb[:,0].sum()}')

  print(a.shape, a.dim())

  # do projectionbias_mask
  lin = torch.nn.Linear(1024, 1024, bias=False, device="cuda", dtype=torch.float16)
  q = lin(a)

  print(q.shape, q.dim())

  # split heads
  p = q.unflatten(-1, [8, 128]).transpose( 1, 2)
  # alternative reshape() calls:
  # p = q.reshape(-1, -1, 8, 128)
  # p = q.reshape( q.shape[0], -1, 1, 8, 128).transpose( 1, 3)
  # p = q.reshape( 4, 32, -1, 1, 8, 128).transpose( 1, -2)
  print(p.shape, p.dim())

  t_start = time.time()
  for _ in range(n_iters):
    with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
      out = torch.nn.functional.scaled_dot_product_attention( p, p, p).transpose( 2, 1)
  a = out.unbind()
  a[0].sum().backward()
  print( f'{(time.time() - t_start) / n_iters}', flush=True)
  print( out.shape) 
  print( out.unbind()[0].shape)
  print( 'Finished func1.\n\n')

func1_compiled = torch.compile( func1)
func1_compiled()