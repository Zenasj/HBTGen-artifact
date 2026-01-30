import torch.jit as jit


@jit.script
def test():
    assert 1 == 1
    return 1


print(test.graph)