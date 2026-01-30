import torch


def fn(inputs, op_inputs_dict):
    res = getattr(inputs, "exponential_")(**op_inputs_dict)
    return res


if __name__ == "__main__":
    inputs = torch.randn(2, 3, 4)
    op_inputs_dict = {'lambd': 10, 'generator': None}
    compl_fn = torch.compile(fn, dynamic=True, backend="eager")
    res = compl_fn(inputs, op_inputs_dict)
    print(res)