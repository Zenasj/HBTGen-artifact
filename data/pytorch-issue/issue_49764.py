import torch

@torch.jit.script
def lstm_net(xs, states, lstm_vars):    
    # type: (Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tensor
    W1, W2 = lstm_vars
    h, c = states
    for x in xs:
        ifgo = torch.cat([x, h, c], dim=1) @ W1
        i, f, g, o = torch.chunk(torch.sigmoid(ifgo), 4, dim=1)
        c = f*c + i*(2.0*g - 1.0) 
        h = o*torch.tanh(c) 
    return torch.sum(h @ W2)

dim_in, dim_hidden, dim_out, batch_size = 10, 10, 10, 10
lstm_vars = [torch.randn(dim_in + 2*dim_hidden, 4*dim_hidden),
             torch.randn(dim_hidden, dim_out)]
[W.requires_grad_(True) for W in lstm_vars]
for num_iter in range(10):
    loss = lstm_net(torch.randn(10, batch_size, dim_in), 
                    torch.unbind(torch.randn(2, batch_size, dim_hidden)), 
                    lstm_vars)
    grads = torch.autograd.grad(loss, lstm_vars, create_graph=True)
    vs = [torch.randn(W.shape) for W in lstm_vars]
    Hvs = torch.autograd.grad(grads, lstm_vars, vs)  
    print(num_iter)

@jit.script
def f(x):
	r = x * torch.ones_like(x) #fails only with local variables
	return torch.unbind(r)[0]
x=torch.zeros(2,16).requires_grad_()
r1 = f(x)
r1.sum().backward()
x.grad = None
print("second pass")
r1 = f(x)
r1.sum().backward()