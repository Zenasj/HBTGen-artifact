import torch
import torch.nn as nn

def hook(module, inputs, output):
    if torch.is_tensor(output):
        if 'module' not in output.grad_fn.metadata:
            output.grad_fn.metadata['module'] = module
        if 'output' not in output.grad_fn.metadata:
            output.grad_fn.metadata['output'] = output
            
def register_forward_hooks(model, hook):
    for name, mod in model.named_modules():
        if not mod._modules: # is a leaf module
            mod.register_forward_hook(hook)    
        
model = nn.Conv2d(10, 10, 3)
register_forward_hooks(model, hook)
inputs = (torch.randn(2,10,32,32),)
output = model(*inputs)
print(output.grad_fn.metadata['module'].kernel_size)
print(output.grad_fn.metadata['output'].shape)