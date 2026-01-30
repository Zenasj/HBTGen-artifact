import torch


def fn(test_args):
    res = torch.gather(**test_args)
    return res


if __name__ == "__main__":
    test_args = {
        "input": torch.tensor([[3.0, -5.0]], dtype=torch.bfloat16, requires_grad=True),
        "dim": 1,
        "index": torch.tensor([[1]]),
    }
    fn = torch.compile(fn)
    res_fwd = fn(test_args)
    bwd_tensor = torch.randn(res_fwd.shape, dtype=torch.bfloat16)
    res_fwd.backward(bwd_tensor)
    print(res_fwd, test_args["input"].grad)