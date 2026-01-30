import torch.nn as nn

import torch
import torch._dynamo


class DummyNNModule(torch.nn.Module):
    def __init__(self, op, inplace_op, inplace_tensor):
        super().__init__()
        self.op = op
        self.inplace_op = inplace_op
        self.inplace_tensor = inplace_tensor

    def forward(self, op_inputs_dict):
        if self.inplace_op:
            self.inplace_tensor = self.inplace_tensor[
                slice(None, -4, None),
                slice(None, -1, None),
                slice(None, -1, None),
                slice(None, -1, None),
            ]
            result = getattr(self.inplace_tensor, self.op)(**op_inputs_dict)
        else:
            result = self.op(**op_inputs_dict)
        return result


class DummyClassExec:
    def __init__(self, op, inplace_op, inplace_tensor, op_params):
        self.op = op
        self.inplace_op = inplace_op
        self.inplace_tensor = inplace_tensor
        self.op_params = op_params

    def run_in_compile(self):
        model = DummyNNModule(
            self.op,
            self.inplace_op,
            self.inplace_tensor,
        )
        model = torch.compile(model, fullgraph=True, backend="eager")

        result = model(self.op_params)
        return result

def run_add_inplace():
    add_exec = DummyClassExec(
        "add_", True, torch.randn([8, 26, 28, 4]), {"other": 0.9, "alpha": 0.4}
    )
    add_exec.run_in_compile()
    print("Add_ compile success!")

def run_bce():
    op_params = {
        "input": torch.randn([2, 9]).uniform_(0, 1),
        "target": torch.randn([2, 9]).uniform_(0, 1),
        "reduction": "mean",
    }
    bce_exec = DummyClassExec(
        torch.nn.functional.binary_cross_entropy, False, None, op_params
    )
    bce_exec.run_in_compile()
    print("BCE compile success!")

if __name__ == "__main__":
    run_add_inplace() # if add inplace not executed then BCE execution fine
    run_bce()