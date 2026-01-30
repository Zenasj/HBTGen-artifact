import torch
import torch._dynamo as dynamo
 
dynamo.reset()
 
def toy_example(inp):
    class A:
        def __init__():
            pass
        def something():
            pass
    return inp + 2
 
compiled_fn = torch.compile(toy_example, backend="inductor")

inp = torch.arange(2)
r1 = compiled_fn(inp)
print(f"r1 = {r1}")