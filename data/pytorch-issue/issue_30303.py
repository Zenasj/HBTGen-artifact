from argparse import ArgumentParser

import torch
from torch import Tensor


def check_jacobian(x: Tensor, weight: Tensor):
    print(f'{"contiguous" if x.is_contiguous() else "incontiguous"} input tensor')

    y1 = x @ weight
    y2 = torch.einsum('...x,xy->...y', [x, weight])
    assert torch.allclose(y1, y2)

    x.grad = None
    y1.backward(torch.eye(2, device=x.device))
    print(f'matmul.jacobian => {x.grad.view(-1)}')

    x.grad = None
    y2.backward(torch.eye(2, device=x.device))
    print(f'einsum.jacobian => {x.grad.view(-1)}')


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--device', type=int, default=-1)
    args = argument_parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    device = torch.device(f'cuda:{args.device}')

    _x = torch.rand(1, 2, device=device)
    x1 = _x.expand(2, -1).requires_grad_()
    x2 = _x.expand(2, -1).contiguous().requires_grad_()
    w = torch.rand(2, 2, device=device)

    check_jacobian(x1, w)
    check_jacobian(x2, w)