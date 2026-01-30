import torch

def g1(args): ...
def g2(args): ...
def f(args):
    g1(args)
    print("graph break")
    g2(args)

with torch._dynamo.config.patch(CONFIG1):
    torch.compile(f)(args)  # graph break means we store compile products on g1, g2 based on CONFIG1

with torch._dynamo.config.patch(CONFIG2):
    torch.compile(g2)(args)  # will now recompile, as g2's (newly created) dynamo context config 
                             # differs from previous compiled products'