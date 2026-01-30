import torch.nn as nn

import torch


class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_dict):
        return torch.nn.functional.embedding(**input_dict)


if __name__ == "__main__":
    input_dict = {
        "input": torch.tensor([[7, 2, 3, 4], [8, 4, 4, 7]]),
        "weight": torch.randn([10,3], dtype=torch.float16, requires_grad=True),
        "padding_idx": 1,
        "scale_grad_by_freq": True,
    }

    model = BasicModule()
    model = torch.compile(model)
    fwd_res = model(input_dict)
    print(fwd_res)