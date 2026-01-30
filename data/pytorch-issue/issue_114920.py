import torch

OpOverloadPacket.__call__(self, {"self": ...})

OpOverloadPacket.__call__(_self, {"self": ...})

In [4]: [x.name for x in torch.ops.aten.reshape.default._schema.arguments]
Out[4]: ['self', 'shape']

In [3]: torch.ops.aten.reshape.default(self=torch.rand(1,2), shape=[2])
Out[3]: tensor([0.5127, 0.3051])