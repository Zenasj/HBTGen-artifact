import torch
import torch._dynamo as dynamo
 
def toy_example(inputs):
    out = inputs[0]
    for inp in filter(lambda x: (x.requires_grad), inputs):
      out = out * inp
    out = out + 2000
    return out
 
input1 = torch.arange(2, dtype=torch.bfloat16)
input2 = torch.arange(2, dtype=torch.bfloat16).requires_grad_(True)
 
compiled_fn = torch.compile(toy_example, backend="inductor")
 
r1 = compiled_fn([input1, input2])
print("r1 = ", r1.cpu())