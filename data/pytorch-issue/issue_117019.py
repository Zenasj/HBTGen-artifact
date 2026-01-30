import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input): 
        input = torch.diag_embed(input=input, dim1=-1,dim2=0,offset=1)        
        return input

x = torch.rand([8, 6, 8, 6, 6, 1], dtype=torch.float64)
model = Model().to(torch.device('cpu'))
eag = model(x)
opt = torch.compile(model.forward)(x)

same_val = torch.allclose(eag.to('cpu'), 
                            opt.to('cpu'), 
                            rtol=1e-3, atol=1e-3, 
                            equal_nan=True)
if same_val == False : 
        raise ValueError('diff value')

@register_decomposition(aten.diag_embed)
@out_wrapper()
def diag_embed(
    t: TensorLikeType,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    # Code A
    #as per the docs, exchanging dims is equivalent to changing the sign of
    # offset
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset
    # Code B
    # convert from negative dims
    rank = t.ndim + 1
    dim1 = utils.canonicalize_dim(rank=rank, idx=dim1)
    dim2 = utils.canonicalize_dim(rank=rank, idx=dim2)
    ...
    # Code C
    # preserve original data, but place 1 at dim1 and move last dim to dim2
    t = t.unsqueeze(dim1).movedim(-1, dim2)
    ...