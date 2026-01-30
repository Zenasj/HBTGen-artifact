import torch
import torch.nested as nested

x = (torch.arange(20).reshape((5,4)) * 1.).requires_grad_()
y = (torch.arange(12).reshape((3,4)) * 10.).requires_grad_()
n = nested.as_nested_tensor([x, y], layout=torch.jagged)
w = torch.tensor([10., 100.], dtype=torch.float).reshape(2,1,1).requires_grad_()

def redmul(n, w):
    nw = nested.as_nested_tensor([dense * w[idx] for idx, dense in enumerate(n.unbind())], layout=torch.jagged)
    loss = 0.
    for dense in nw.unbind():
        loss = loss + dense.sum()
    return loss

mul = torch.compile(redmul) ### No error without compile
loss = mul(n, w)
loss.backward()