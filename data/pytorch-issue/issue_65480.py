import torch

def foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
    return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))

@torch.jit.script
def jitted_foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
    return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))

x = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
y = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
W = torch.randn(16, 1, dtype=torch.float32, device='cuda:0', requires_grad=True)
W.data /= 4
print('1st launch - foo(...).grad_fn is None = {}'.format(foo(x, y, W).grad_fn is None)) # outputs False, which is what's expected
print('2nd launch - foo(...).grad_fn is None = {}'.format(foo(x, y, W).grad_fn is None)) # outputs False, again as expected
print('1st launch - jitted_foo(...).grad_fn is None = {}'.format(jitted_foo(x, y, W).grad_fn is None)) # outputs False, again as expected
print('2nd launch - jitted_foo(...).grad_fn is None = {}'.format(jitted_foo(x, y, W).grad_fn is None)) # outputs True, meaning the graph is lost

traced_foo = torch.jit.trace(foo, (x, y, W.detach()))
print('1st launch - traced_foo(...).grad_fn is None = {}'.format(traced_foo(x, y, W).grad_fn is None)) # outputs False, as expected
print('2nd launch - traced_foo(...).grad_fn is None = {}'.format(traced_foo(x, y, W).grad_fn is None)) # outputs False, as expected

W = torch.randn(16, 1, dtype=torch.float32, device='cuda:0', requires_grad=True)/4

W = torch.randn(16, 1, dtype=torch.float32, device='cuda:0', requires_grad=True)
W.data /= 4

import torch

def foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
    return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))

@torch.jit.script
def jitted_foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
    return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))

x = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
y = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
W = torch.randn(16, 1, dtype=torch.float32, device='cuda:0', requires_grad=True)
W.data /= 4
# BEGIN WORKAROUND FOR jitted_foo
with torch.no_grad():
    jitted_foo(x, y, W)
# END WORKAROUND FOR jitted_foo
print('1st launch - foo(...).grad_fn is None = {}'.format(foo(x, y, W).grad_fn is None)) # outputs False, which is what's expected
print('2nd launch - foo(...).grad_fn is None = {}'.format(foo(x, y, W).grad_fn is None)) # outputs False, again as expected
print('1st launch - jitted_foo(...).grad_fn is None = {}'.format(jitted_foo(x, y, W).grad_fn is None)) # outputs False, again as expected
print('2nd launch - jitted_foo(...).grad_fn is None = {}'.format(jitted_foo(x, y, W).grad_fn is None)) # NEW: now outputs False, meaning the workaround works
traced_foo = torch.jit.trace(foo, (x, y, W.detach()))
print('1st launch - traced_foo(...).grad_fn is None = {}'.format(traced_foo(x, y, W).grad_fn is None)) # outputs False, as expected
print('2nd launch - traced_foo(...).grad_fn is None = {}'.format(traced_foo(x, y, W).grad_fn is None)) # outputs False, as expected