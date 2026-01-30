import torch

def sequential_scan(bin_op, x):
    s, x_tail = x[..., 0], x[..., 1:]
    outputs = [s]
    for xs_i in torch.unbind(x_tail, dim=-1):
        s = bin_op(s, xs_i)
        outputs.append(s)

    output = torch.stack(outputs, dim=-1)
    return output

def test_scan(L):
    sum_op = lambda x, y: x + y
    a = 1 + torch.arange(L)

    csum = sequential_scan(sum_op, a)
    print(csum)

    scan_opt = torch.compile(sequential_scan)
    csum = scan_opt(sum_op, a)
    print(csum)
    print()

if __name__ == "__main__":
    test_scan(4)
    test_scan(1)