import torch
import torch.nn as nn
import spconv.pytorch as spconv
from voydet.architect.cnn.layers import build_norm_layer

class MLPModel(nn.Module):
    def __init__(self, sparse_shape, 
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()
        self.sparse_shape = sparse_shape
        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(3,
                                16,
                                3,
                                padding=1,
                                bias=False,
                                indice_key='subm1'),
            build_norm_layer(norm_cfg, 16), 
            nn.ReLU())
        
        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(16,
                                16,
                                3,
                                padding=1,
                                bias=False,
                                indice_key='subm1'),
            build_norm_layer(norm_cfg, 16), 
            nn.ReLU())

    def forward(self, features, coords, batch_size=1):
        x = spconv.SparseConvTensor(
            features=features,
            indices=coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


sparse_shape = [40, 100, 100]
model = MLPModel(sparse_shape=sparse_shape).cuda()
features = torch.rand(200, 3).cuda()
coords = [(torch.rand(200) * 0).int(), (torch.rand(200) * 40).int(), (torch.rand(200) * 100).int(), (torch.rand(200) * 100).int()]
coords = torch.stack(coords, dim=1).cuda()
x = model(features, coords)
model.eval()
dummy_inputs = (features, coords)
name_inputs = ('features', 'coords')
with torch.no_grad():
    torch.onnx.export(model,
                        dummy_inputs,
                        'asdasd.onnx',
                        opset_version=11,
                        input_names=name_inputs,
                        output_names='output')