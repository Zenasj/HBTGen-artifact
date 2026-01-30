import torch

x = x + 1
torch._dynamo.eval_frame.raise_sigtrap();
# can breakpoint on ceval.c:CALL to breakpoint the `sin` call in C.
x = torch.sin(x)