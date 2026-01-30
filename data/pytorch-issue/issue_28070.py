import torch

input = torch.randn([1, 3, 224, 224], dtype=torch.float)
input = torch.quantize_per_tensor(input, scale=1e-3, zero_point=0, dtype=torch.qint32)  # This is just an example
out = quant_model(input)

def __init__(self):
    ...
    self.quant = QuantStub() # Quantize stub module, before calibration, this is same as an observer, it will be swapped as nnq.Quantize in convert.
    ...
def forward(self, x):
    x = self.quant(x)
    ...