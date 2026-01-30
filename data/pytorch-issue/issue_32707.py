import torch as th
from concurrent.futures import ProcessPoolExecutor

def run():
    x = th.tensor([1.], requires_grad=True)
    for _ in range(100):
        loss = th.sum((x + 1)**2)
        loss.backward()
        x.data -= 0.01 * x.grad
        x.grad.zero_()
    return x[0].item()
        
with ProcessPoolExecutor(1) as pool:
    r = pool.submit(run)
    print("before it works", r.result())
print("calling run in main process", run())
print("Now it breaks, but only in jupyter notebook")
with ProcessPoolExecutor(1) as pool:
    r = pool.submit(run)
    r.result()