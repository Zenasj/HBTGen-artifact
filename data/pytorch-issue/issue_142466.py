import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, operator):
        super(Model, self).__init__()
        self.op = operator

    def forward(self, x):
        x = self.op(x)
        return x


def run_test(operator, dim, device):
    op_inst = eval(f"torch.nn.{operator}{dim}d(in_channels=1, out_channels=0, kernel_size={tuple([1] * dim)})")
    model = Model(op_inst).to(device)
    x = torch.randn([1] * (dim + 1)).to(device)

    try:
        y = model(x)
        print(f"success: {y}")
    except Exception as e:
        print(e)


run_test("ConvTranspose", 1, "cpu")  # non-empty 2D or 4D weight tensor expected, but got: [1, 0, 1, 1]
run_test("ConvTranspose", 1, "cuda")  # success: tensor([], device='cuda:0', size=(0, 1), grad_fn=<SqueezeBackward1>)
run_test("ConvTranspose", 2, "cpu")  # non-empty 2D or 4D weight tensor expected, but got: [1, 0, 1, 1]
run_test("ConvTranspose", 2, "cuda")  # success: tensor([], device='cuda:0', size=(0, 1, 1), grad_fn=<SqueezeBackward1>)
run_test("ConvTranspose", 3, "cpu")  # non-empty 5D (n_output_plane x n_input_plane x kernel_depth x kernel_height x kernel_width) tensor expected for weight, but got: [1, 0, 1, 1, 1]
run_test("ConvTranspose", 3, "cuda")  # success: tensor([], device='cuda:0', size=(0, 1, 1, 1), grad_fn=<SqueezeBackward1>)