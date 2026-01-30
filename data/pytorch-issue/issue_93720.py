import torch

def test_capture(x):
    try:
       x = x + 1  # should be captured fine
       assert False
    except:
       x = x + 2  # won't be captured
       return x

torch.compile(test_capture, backend="eager", fullgraph=False)(torch.rand(10))