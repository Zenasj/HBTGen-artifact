import torch

def closure_repro():
    def outer(a):
        return a + 1

    def indirect(x):
        return direct(x)

    def direct(x):
        def inner(b):
            return b + 2

        def deep(c):
            d = outer(c)
            return inner(d)

        return deep(x)

    dynamo.export(indirect, torch.randn(3))  # changing this to export `direct` instead works!


closure_repro()