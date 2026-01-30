import torch

def test_function(func, inputs):
    try:
        func(*inputs)
        print("No error with function {}".format(func.__name__))
    except Exception as e:
        print("Exception message: {}\nGiven argument names: {}".format(e, inputs))

if __name__ == '__main__':
    cases = [
             ('addmm', [torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)]),
             ('addmv', [torch.randn(2),    torch.randn(2, 3), torch.randn(3)]),
             ('addr',  [torch.zeros(3, 2), torch.arange(1., 4.), torch.arange(1., 3.)]),
             ('baddbmm', [torch.randn(3, 2, 4), torch.randn(3, 2, 3), torch.randn(3, 3, 4)]),
             ('bmm', [torch.randn(3, 2, 3), torch.randn(3, 3, 4)]),
             ('dot', [torch.randn(10), torch.randn(10)]),
             ('ger', [torch.randn(2), torch.randn(4)]),
             ('matmul', [torch.randn(3, 3), torch.randn(3, 4)]),
             ('mm', [torch.randn(3, 3), torch.randn(3, 4)]),
             ('mv', [torch.randn(3, 3), torch.randn(3)])
            ]
    for case in cases:
        test_function(getattr(torch, case[0]), case[1])