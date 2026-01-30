import torch
from torch.optim import Adam, Adadelta, Adamax, AdamW
from torch import _dynamo
import gc

def optimizer_step(optimizer):
    def f():
        optimizer.step()
    f()

def pt2_optimizer_step(optimizer):
    @torch.compile()
    def f():
        optimizer.step()
    f()

Optims = [Adam, Adadelta, Adamax, AdamW, Adam, Adadelta, Adamax, AdamW]
parameters = [torch.rand(1000000, 64, dtype=torch.float32, device='cuda') for _ in range(10)]
for p in parameters:
    p.grad = torch.rand_like(p)

print('CUDA memory taken up by params and grads: ', torch.cuda.memory_allocated())
print('\n------------------------------------------------------------------------------')

for O in Optims:
    optim = O(parameters)
    optimizer_step(optim)
    print('CUDA memory after optim step: ', torch.cuda.memory_allocated())

print('\n------------------------------------------------------------------------------')
print('Note the above behavior shows constant memory! The following will show that CUDA memory gets used up.')

for O in Optims:
    optim = O(parameters)
    pt2_optimizer_step(optim)
    # torch.cuda.empty_cache()     # doesn't help
    print('CUDA memory after torch compiled optim step: ', torch.cuda.memory_allocated())
_dynamo.reset() 

print('\n------------------------------------------------------------------------------')
print('Let us try that again but with gc.collect()')

for O in Optims:
    optim = O(parameters)
    pt2_optimizer_step(optim)
    # torch.cuda.empty_cache()     # doesn't help
    gc.collect()                 # doesn't help
    print('CUDA memory after torch compiled optim step: ', torch.cuda.memory_allocated())
_dynamo.reset() 

print('\n------------------------------------------------------------------------------')
print('Let us try that again but with both gc.collect() and _dynamo.reset()')

for O in Optims:
    optim = O(parameters)
    pt2_optimizer_step(optim)
    # torch.cuda.empty_cache()     # doesn't help
    gc.collect()                 # doesn't help when alone
    _dynamo.reset()              # actually causes OOMs when alone, see below example
    print('CUDA memory after torch compiled optim step: ', torch.cuda.memory_allocated())
_dynamo.reset()

print('\n------------------------------------------------------------------------------')
print('Let us try that again but with _dynamo.reset()')

for O in Optims:
    optim = O(parameters)
    pt2_optimizer_step(optim)
    # torch.cuda.empty_cache()     # doesn't help
    # gc.collect()                 # doesn't help
    _dynamo.reset()                # WILL OOM
    print('CUDA memory after torch compiled optim step: ', torch.cuda.memory_allocated())