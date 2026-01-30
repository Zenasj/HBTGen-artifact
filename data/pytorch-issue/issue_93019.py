import torch
import torch._dynamo

def ones():
  return torch.ones([4], device="cuda")

ten = torch.rand([4], device="cuda")
foo1 = lambda x: x + ten
foo2 = lambda x: x + ten.sum()

fn1 = lambda: foo1(ones())
fn2 = lambda: foo2(ones())

fn2_opt = torch._dynamo.optimize("inductor")(fn2)
fn1_opt = torch._dynamo.optimize("inductor")(fn1)
# Maybe not important, but switching the order of calling fn1_opt(), fn2_opt() below no longer segfaults
fn1_opt()
fn2_opt()