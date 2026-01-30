import torch
import torch.nn as nn

f(batch)[:k] == f(batch[:k])

f(batch)[:batch] == f(batch[:batch])

def check(batch, input_dim, output_dim, topk=1):
  """
  The output of f(batch)[:topk] is different than f(batch[:topk]).
  
  This functions tries to check that for different input_dimension and output_dimension.
  
  f here is a linear layer (or more barebone version i.e a matmul)
  """
  
  # input of batch x input_dim
  x = torch.randn(batch, input_dim)
  # weight matrix of output_dim x input_dim (used for matmul to replicate linear layer results)
  w = torch.randn(output_dim, input_dim)
  # linear layer that takes input of input_dim and spits out output_dim.. (set to bias to false for simplicity)
  f = nn.Linear(input_dim, output_dim, bias=False)
  
  # this is a boolean to compare the output of linear layer
  out1 = (f(x)[:topk] == f(x[:topk])).all()
  # this is a boolean to compare the output of matmul layer
  out2 = (x.matmul(w.t())[:topk] == x[:topk].matmul(w.t())).all()
  
  # the output should be the same for both linear and matmul
  return (out1 & out2).item()

check(batch=300, input_dim=3, output_dim=100, topk=300) # True
check(batch=300, input_dim=3, output_dim=100, topk=1)   # False

f(batch)[:batch] == f(batch[:batch])

f(batch)[:k] == f(batch[:k])