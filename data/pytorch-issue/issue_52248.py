class NaNWhere(torch.autograd.Function):
    '''                                 
    An adaptation of torch.where to situations like
    output = torch.where(torch.isfinite(f1(x)), f1(x), f2(x))
    where the point of torch.where is to mask out invalid application
    of f1 to some elements of x and replace them with fallback values
    provided by f2(x).                                                
    '''                
    @staticmethod
    def forward(ctx, x, f1, f2, mask_fn=torch.isfinite):
        x_1 = x.detach().clone().requires_grad_(True)   
        x_2 = x.detach().clone().requires_grad_(True)
                                                     
        with torch.enable_grad():
            y_1 = f1(x_1)        
            y_2 = f2(x_2)
                         
        mask = mask_fn(y_1)
                           
        ctx.save_for_backward(mask)
        ctx.x_1 = x_1              
        ctx.x_2 = x_2
        ctx.y_1 = y_1 
        ctx.y_2 = y_2
                     
        return torch.where(mask, y_1, y_2)
                                          
    @staticmethod
    def backward(ctx, gout):
        mask, = ctx.saved_tensors
                                 
        torch.autograd.backward([ctx.y_1, ctx.y_2], [gout, gout])
        gin = torch.where(mask, ctx.x_1.grad, ctx.x_2.grad)      
                                                           
        return gin, None, None
                               
nan_where = NaNWhere.apply

import torch

def base_where_wrapper(x, f1, f2):
    '''
    This will lead to NaN gradients.
    '''
    y1 = f1(x)
    y2 = f2(x)

    return torch.where(torch.isfinite(y1), y1, y2)

class NaNWhere(torch.autograd.Function):
    '''
    An adaptation of torch.where to situations like
    output = torch.where(torch.isfinite(f1(x)), f1(x), f2(x))
    where the point of torch.where is to mask out invalid application
    of f1 to some elements of x and replace them with fallback values
    provided by f2(x).
    '''
    @staticmethod
    def forward(ctx, x, f1, f2, mask_fn=torch.isfinite):
        x_1 = x.detach().clone().requires_grad_(True)
        x_2 = x.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            y_1 = f1(x_1)
            y_2 = f2(x_2)

        mask = mask_fn(y_1)

        ctx.save_for_backward(mask)
        ctx.x_1 = x_1
        ctx.x_2 = x_2
        ctx.y_1 = y_1
        ctx.y_2 = y_2

        return torch.where(mask, y_1, y_2)

    @staticmethod
    def backward(ctx, gout):
        mask, = ctx.saved_tensors

        torch.autograd.backward([ctx.y_1, ctx.y_2], [gout, gout])
        gin = torch.where(mask, ctx.x_1.grad, ctx.x_2.grad)

        return gin, None, None

nan_where = NaNWhere.apply

def test_case(where_impl):
    x = torch.linspace(-5, 5, 10, requires_grad=True) # includes negative numbers
    
    def f1(x):
        return x.sqrt() # sqrt will output NaN for negative numbers

    def f2(x):
        return (-x).sqrt()

    selected = where_impl(x, f1, f2)
    
    selected.sum().backward()

    return torch.isfinite(x.grad).all()

if __name__ == '__main__':
    print(test_case(base_where_wrapper))
    print(test_case(nan_where))