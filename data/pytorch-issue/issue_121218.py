import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.NLLLoss()

    def forward(self, ifm_param):
        result = self.op(**ifm_param)
        return result


if __name__ == "__main__":
    torch.manual_seed(0)
    compiled_model = torch.compile(Model())
    ifm_params = [
        {
            "input": torch.randn([16, 24, 8], requires_grad=True),
            "target": torch.randint(low=0, high=10, size=[16, 8]),
        },
        {
            "input": torch.randn([16, 13068, 8], requires_grad=True),
            "target": torch.randint(low=0, high=50, size=[16, 8]),
        },
    ]
    for i, ifm_param in enumerate(ifm_params):
        print(f"running iter: {i}")
        result = compiled_model(ifm_param)
print("Successfully executed!")