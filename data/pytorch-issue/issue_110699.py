import torch

def dummy_fn(op_inputs_dict):
    return torch.index_select(**op_inputs_dict)

if __name__ == "__main__":
    torch.manual_seed(10)
    op_inputs_dict = {
        "input": torch.randn([8, 16]),
        "dim": 0,
        "index": torch.tensor([2, 1, 6, 7, 3, 1, 7, 5, 6, 7]),
        "out": torch.empty([10,16]), # should be ([8,16]),
    }
    res = dummy_fn(op_inputs_dict)
    print(res, res.shape)
    model = torch.compile(dummy_fn, backend="eager")
    res = model(op_inputs_dict)
    print(res)