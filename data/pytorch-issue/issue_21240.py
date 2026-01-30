import torch

@torch.jit.script
def fib(x):
    # type: (int) -> int
    prev = 1
    v = 1
    for i in range(0, x):
        save = v
        v = v + prev
        prev = save
        # print(v)

    #print("Done")
    return v



fib(10000000)