import torch

def fn(x):
    return x.argmax(0)

for i in range(100):
    x = torch.randint(-100, 100, (20, 20), dtype=torch.int16)

    cpu = fn(x)
    gpu = fn(x.cuda()).cpu()
    compiled = torch.compile(fn)
    comp = compiled(x.cuda()).cpu()

    print(i)

    assert torch.allclose(cpu, gpu), '\n'.join(['', 'cpu', str(cpu), 'gpu', str(gpu)])
    assert torch.allclose(cpu, comp), '\n'.join(['', 'cpu', str(cpu), 'comp', str(comp)]) # may fail sometimes