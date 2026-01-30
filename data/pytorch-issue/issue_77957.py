3
import torch
from torch.testing import assert_close
a = torch.ones(1)
b = torch.zeros(1)
inf = a/b
nan = b/b
cpu = torch.device('cpu')
mps = torch.device('mps')
print ("mps is ok with having nan and inf", inf.to(mps), nan.to(mps))
print ("assert_close on CPU")
try:
    assert_close(a.to(cpu), inf.to(cpu))
except Exception as er:
    print (er)
print ("assert_close on MPS")
try:
    assert_close(a.to(mps), inf.to(mps)) # bug is here
except Exception as er:
    print (er)