import torch
def test(device, accumulate):
    x = torch.randn([2, 3, 4], device=device)
    i1 = torch.as_tensor([0, 1], device=device)
    i2 = torch.as_tensor([True, True, False], device=device)
    v = torch.randn([], device=device)
    torch.ops.aten.index_put_(x, [i1, i2, None], v, accumulate)
    print(f"Test on {device}(accumulate={accumulate}) passed")

test('cpu', False)
test('cuda', False)
test('cpu', True)
test('cuda', True)

import torch
def test(device, accumulate):
    x = torch.randn([2, 3, 4], device=device)
    i1 = torch.as_tensor([0, 1], device=device)
    i2 = i1 
    v = torch.randn([], device=device)
    torch.ops.aten.index_put_(x, [i1, i2, None], v, accumulate)
    print(f"Test on {device}(accumulate={accumulate}) passed")

test('cpu', False)
test('cuda', False)
test('cpu', True)
test('cuda', True)