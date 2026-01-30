import torch

@parse_args('v')
def hardsigmoid(g, self):
    input_ = g.op("Div", self, g.op('Constant', value_t=torch.tensor(3, dtype=torch.float)))
    hardsigmoid_ = sym_help._hardtanh_helper(
        g,
        input_,
        g.op('Constant', value_t=torch.tensor(-1, dtype=torch.float)),
        g.op('Constant', value_t=torch.tensor(1, dtype=torch.float))
    )
    hardsigmoid_ = g.op("Add", hardsigmoid_, g.op('Constant', value_t=torch.tensor(1, dtype=torch.float)))
    return g.op("Div", hardsigmoid_, g.op('Constant', value_t=torch.tensor(2, dtype=torch.float)))