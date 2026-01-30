py
import torch
import torch.nn as nn

torch.manual_seed(420)

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.fc(x)
        return x
input_tensor = torch.randn(1, 10).to(torch.bfloat16)

func = MyModel().to('cpu')

print(func(input_tensor))
# tensor([[-0.6367]], dtype=torch.bfloat16, grad_fn=<AddmmBackward0>)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    print(jit_func(input_tensor))
# RuntimeError: mkldnn_reorder_linear_weight: bf16 path needs the cpu support avx512bw, avx512vl and avx512d