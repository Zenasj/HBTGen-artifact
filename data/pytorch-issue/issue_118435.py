import torch

def compiler_fn(gm):
    def inner(gm_, example_inputs_):
        return inductor.compile(gm_, example_inputs_)
    return torch.compile(gm, fullgraph=True, backend=inner)

def hook(param):
    param.grad *= 2

x = torch.ones(10)
x.requires_grad = True
def run(input):
    return x * input

x.register_post_accumulate_grad_hook(hook)
with compiled_autograd.enable(compiler_fn):
    for i in range(5):
        run(input).sum().backward()
        # Mimic optimizer.zero_grad() to clear the gradient
        print(x.grad)
        x.grad = None