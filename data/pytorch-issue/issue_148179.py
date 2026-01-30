import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.linear = torch.nn.Linear(K, N)

    def forward(self, input):
        inputs = []
        mf = [1, 2]
        if mf != -1:
            inputs.append(input)
        return self.linear(inputs[0])

if __name__ == "__main__":

    with torch.no_grad():
        M = 1024
        K = 1024
        N = 1024
        input = torch.randn(M, K)
        m = Model(K, N).eval()

        example_inputs = (input,)
        exported_model = torch.export.export_for_training(
            m,
            example_inputs,
        )
        c_m = exported_model.module()
        res = c_m(input)