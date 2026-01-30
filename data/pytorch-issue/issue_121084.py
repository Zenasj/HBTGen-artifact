import torch
from torch.func import grad, hessian

print('torch version:',torch.__version__)

class Parabola(torch.autograd.Function):
    
    generate_vmap_rule=True
    
    
    @staticmethod
    def forward(x, abc):
        a, b, c = abc
        y = a*x.square() + b*x + c
        dydx = 2*a*x + b
        return y, dydx
    
    @staticmethod
    def setup_context(ctx, inputs, outputs): # I was surpised to see this is called twice for every invocation of forward   
        y, dydx = outputs
        ctx.save_for_backward(dydx)
    
    @staticmethod
    def backward(ctx, Dy, dummy):
        dydx, = ctx.saved_tensors        # this works as expected
        return Dy*dydx, None
    
    @staticmethod
    def jvp(ctx, Dx, dummy):
        dydx, = ctx.saved_tensors # crashes here, becasause RHS is empty
        return Dx*dydx, None

abc = torch.randn(3)    
    

def f0(x):
    a, b, c = abc
    return a*x.square() + b*x + c

def f(x): 
    y, dydx = Parabola.apply(x, abc)
    return y

x = torch.tensor(torch.pi)

print('\n\ntesting grad:')
print(grad(f0)(x))
print(grad(f)(x))

print('\n\ntesting hessian:')
print(hessian(f0)(x))
print(hessian(f)(x))

import torch
from torch.func import grad
from torch.autograd import forward_ad as fwAD


class Parabola(torch.autograd.Function):
    
    
    @staticmethod
    def forward(x, abc):
        a, b, c = abc
        y = a*x.square() + b*x + c
        dydx = 2*a*x + b
        return y, dydx
    
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        y, dydx = outputs
        ctx.save_for_backward(dydx)
        #ctx.dydx = dydx              # this also doesn't work
    
    @staticmethod
    def backward(ctx, Dy, dummy):
        dydx, = ctx.saved_tensors
        #dydx = ctx.dydx              # this also doesn't work
        return Dy*dydx, None
    
    @staticmethod
    def jvp(ctx, Dx, dummy):
        dydx, = ctx.saved_tensors
        #dydx = ctx.dydx              # this also doesn't work
        return Dx*dydx, None

abc = torch.randn(3)    
    

def f0(x):
    """
    Ordinary pytorch function for control.
    """
    a, b, c = abc
    return a*x.square() + b*x + c

def f(x): 
    """
    My custom function.
    """
    y, dydx = Parabola.apply(x, abc)
    return y

x = torch.tensor(torch.pi)

print('\n\ntesting grad:')
a,b,c = abc
print('control  :', 2*a*x+b)
print('rev ad f0:', grad(f0)(x))
print('rev ad f1:',grad(f)(x))

print('\n\ntesting forward:')
dx = torch.tensor(1/torch.pi)
print('control: jvp =',grad(f0)(x)*dx)

with fwAD.dual_level():
    x_dx = fwAD.make_dual(x, dx)
    y, dy = fwAD.unpack_dual(f0(x_dx))
print('fwd f0 : jvp =',dy)

with fwAD.dual_level():
    x_dx = fwAD.make_dual(x, dx)
    y, dy = fwAD.unpack_dual(f(x_dx))
print('fwd f  : jvp =',dy)