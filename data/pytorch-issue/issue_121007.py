import torch

@torch.compile
def f(x):
    return x.view(x.shape)

x = torch.ones(1)

# Warmup
for i in range(5):
    out = f(x)

number = 1_000_000
repeat = 5
timings = timeit.repeat("f(x)", number=number, repeat=repeat, globals=globals())
print([f"{t:.6f}" for t in timings])

def fn():
    def f(a):
        return a.view(-1)
    fopt = aot_function(f, fw_compiler=nop)
    inp = torch.rand(10, requires_grad=True)
    return fopt(inp)

torch._dynamo.optimize("eager")(fn)()