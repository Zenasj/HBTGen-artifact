import torch
def test_foreach_copy():
    h1 = [torch.randn(1,2), torch.randn(1,3)]
    h2 = [torch.randn(1,2), torch.randn(1,3)]
    def fn(h1, h2):
        return torch.ops.aten._foreach_copy(h1, h2)
    cpu_result = fn(h1, h2)
    print(cpu_result)

    fn = torch.compile(fn)
    h1[0] = h1[0].to('cuda')
    h1[1] = h1[1].to('cuda')

    test_cuda = fn(h1, h2)

    print("cuda result ", test_cuda[0])
    print("cuda result ", test_cuda[1])
test_foreach_copy()