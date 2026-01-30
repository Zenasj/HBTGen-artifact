import torch

def invert(x):
    bound = 255.0
    return bound - x # Overflow happens here due to -x when uint8

x=torch.randint(0, 256, (10, 10), dtype=torch.uint8)

script_invert = torch.jit.script(invert)
invert(x).equal(script_invert(x)) # False

def invert2(x):
    bound = 255.0
    return -(x - bound) # Avoids underflow because x is casted to float before negated


script_invert2 = torch.jit.script(invert2)
invert2(x).equal(script_invert2(x)) # True