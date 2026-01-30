import torch.nn as nn

# t.py
import torch


TRIGGER_SEGFAULT = True

def no_op(*args, **kwargs):
    return []

def compiler_fn(gm):
    if TRIGGER_SEGFAULT:
        return torch.compile(gm, backend="eager")
    else:
        return no_op

def main():
    def inner():
        x = torch.randn(1000, 3000)#, device="cuda")
        w = torch.randn(1000, 3000, requires_grad=True)#, device="cuda")

        def model(i):
            return torch.nn.functional.linear(i, w)

        out = model(x)
        loss = out.sum()
        with torch._dynamo.compiled_autograd.enable(compiler_fn):
            loss.backward()
        print(w.grad)

    inner()
    print("resetting dynamo")
    torch._dynamo.reset()
    inner()


if __name__ == "__main__":
    main()