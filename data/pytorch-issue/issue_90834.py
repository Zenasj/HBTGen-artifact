import torch
from torch.distributions import Normal, Independent

def fn(tensor):
    normal = Normal(tensor, torch.tensor(1))
    independent = Independent(normal, 1)
    return independent.log_prob(tensor) # also for rsample()

a = torch.rand(5, 2)
fn(a)
fn_compiled = torch.compile(fn)
fn_compiled(a)

inspect.getattr_static

UserDefinedObjectVariable

function

self

inspect.getattr_static