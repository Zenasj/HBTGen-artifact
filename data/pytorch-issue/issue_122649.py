import torch

class X:
    _OUTPUTS = { "a": (torch.tensor(1), ), "b": (torch.tensor(2), )}
    @property
    def outputs(self):
        return self._OUTPUTS["a"]
def func(x):
    return x.outputs[0] + 1
eo = torch._dynamo.export(func)(X())