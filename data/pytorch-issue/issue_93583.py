import torch

torch.cuda.memory_allocated()

def fn(a, s):
    x = torch.rand([s, s, s, s], device="cuda")
    return a, x

opt_fn = torch._dynamo.optimize("inductor")(fn)

for i in range(5):
    try:
        res = opt_fn(torch.rand([100, 100], device="cuda"), int(1000/(10**i)))
        print(res[0].shape)
    except RuntimeError as e:
        print(e)