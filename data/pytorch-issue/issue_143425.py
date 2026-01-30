# main file
import torch
import other_file

z = 1
k = 2

def create_fn():
    def fn(x):
        global k
        k = 100
        return x.sin()
    return fn

@torch.compile(backend="eager", fullgraph=True)
def foo(x):
    fn = create_fn()
    global z
    other_file.run_fn(fn, x)
    z = k + 10  # k is not updated to 100

x = torch.randn(2, 3)
foo(x)
print(f'{z=} - {k=}')
assert z == 110
assert k == 100

# second file
def run_fn(fn, x):
    fn(x)