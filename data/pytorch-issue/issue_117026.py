import torch
from monai.data import MetaTensor

torch._dynamo.config.traceable_tensor_subclasses.update({MetaTensor})


t = torch.tensor([1,2,3])
affine = torch.as_tensor([[2,0,0,0],
                            [0,2,0,0],
                            [0,0,2,0],
                            [0,0,0,1]], dtype=torch.float64)
meta = {"some": "info"}
x = MetaTensor(t, affine=affine, meta=meta)


def model(x):
    y = x + 1
    z = x * y
    return z + y - x


cmodel = torch.compile(model, fullgraph=True, backend="eager")

out = cmodel(x)

assert isinstance(out, MetaTensor), type(out)
assert torch.all(out == model(x))
assert out.meta["some"] == "info"
assert torch.all(out.affine == affine)