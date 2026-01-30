import torch.nn as nn

import torch

class IndexFillModel(torch.nn. Module) :
    def __init__ (self, dim_value):
        super().__init__()
        self.dim = torch.tensor(dim_value)

    def forward(self, x, index):
        print(f"in index_fill_, x={x}, dim={self.dim}, index={index}")
        return x.index_fill_(self.dim, index, -1)

model = IndexFillModel(0)
model.eval()
index = torch.tensor([1])
x = torch.tensor([4, 5, 6], dtype=torch.float)
print(f"x={x.shape}, index={index.shape}")
output = model(x.clone(), index)
print(f"model({x}, {index}) ={output}")
onnx_path=f"model.onnx"
torch.onnx.export(model, (x.clone(), index), f=onnx_path)