import torch.nn as nn

dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="mean")

import torch
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 

    def forward(self, args): 
        x, index, input = args
        y_max = input.scatter_reduce(0, index, x, reduce="amax") 
        y_sum = input.scatter_reduce(0, index, x, reduce="sum") 
        y_min = input.scatter_reduce(0, index, x, reduce="amin") 
        y_mul = input.scatter_reduce(0, index, x, reduce="prod") 
        return y_max, y_sum, y_min, y_mul 

model = Model() 
model.eval() 

src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) 
index = torch.tensor([0, 1, 0, 1, 2, 1]) 
input = torch.tensor([1.0, 2.0, 3.0, 8.0]) 

torch.onnx.export(
    model, 
    [src,index,input], 
    f"test.onnx",
    input_names='x,index,input'.split(','),
    opset_version=18,
    verbose=True,
)