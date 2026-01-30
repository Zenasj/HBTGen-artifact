import torch
from monai.networks.nets import VNet

model = VNet(spatial_dims=3, in_channels=1, out_channels=3, dropout_dim=3)
model = model.eval().to('cuda')
data = torch.randn(1,1,32,32,32).to("cuda")

export_output = torch.onnx.dynamo_export(
    model,
    data,
)
export_output.save('Clara_VNet_dynamo.onnx')