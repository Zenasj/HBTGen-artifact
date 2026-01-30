import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1) # if use .half(), result of testcase 1 and 2 will be reversed

    def forward(self, x):
        return self.linear(x)


x = torch.randn(1, 1, dtype=torch.float16)
y = torch.randn(1, 1)

m = Model().eval()
c_m = torch.compile(m)


def run_test(test_case, model, input_tensor):
    print(test_case)
    try:
        output = model(input_tensor)
        print("Success")
    except Exception as e:
        error_type = type(e).__name__
        print(f"{error_type}: {e}")


run_test("testcase 1", m, x)   # RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
run_test("testcase 2", m, y)   # Success
run_test("testcase 3", c_m, x) # Success. If use Conv2d in Model, here will raise Error. In addition, if on CUDA, here will also raise Error
run_test("testcase 4", c_m, y) # Success