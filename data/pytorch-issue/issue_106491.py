import torch

def test_inline_closure_not_loaded_by_parent(self):
    def outer(a):
        return a + 1

    def indirect(x):
        return direct(x)

    def direct(x):
        def deep2(c):
            return outer(c)

        def deep(c):
            return deep2(c)

        return deep(x)

    x = torch.randn(3)
    eager = indirect(x)
    counter = CompileCounter()
    compiled = torch._dynamo.optimize(counter)(indirect)(x)