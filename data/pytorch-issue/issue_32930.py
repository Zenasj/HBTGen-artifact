import torch
import torch.nn as nn


class ABC(nn.Module):
    def __init__(self):
        super(ABC, self).__init__()
        self.l = ReturnTuple()

    def forward(self, x):
        a, b, c = self.l(x)
        # use different ops to distinguish tuple elements
        # the trace should show that a goes to relu and c to softmax
        return a.relu(), b.sigmoid(), c.softmax(1)


class CBA(nn.Module):
    def __init__(self):
        super(CBA, self).__init__()
        self.l = ReturnTuple()

    def forward(self, x):
        a, b, c = self.l(x)
        # use different ops to distinguish tuple elements
        # the trace should show that c goes to relu and a to softmax
        # but looks exactly the same as above
        return c.relu(), b.sigmoid(), a.softmax(1)


class ReturnTuple(nn.Module):
    def forward(self, x):
        # return a tuple
        return x * 3, x * 4, x * 5


if __name__ == "__main__":
    sample = torch.randn(10, 10)
    traced_ABC = torch.jit.trace(ABC(), sample)
    traced_CBA = torch.jit.trace(CBA(), sample)
    scripted_CAB = torch.jit.script(CAB())

    print("ABC trace")
    print(str(traced_ABC.graph))
    print("CBA trace")
    print(str(traced_CBA.graph))
    print("CBA scripted")
    print(str(scripted_CAB.graph))

torch._C._jit_set_inline_everything_mode(True)
traced_CBA = torch.jit.trace(CBA(), (sample,), check_trace=False)
print(traced_CBA.graph)