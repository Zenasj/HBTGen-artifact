import torch
import torch.nn as nn

class BugDemonstrator0(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, features, x): 
        dtype = torch.float
        slice_1 = x[:, 3]
        slice_2 = x[:, 2]
        y1 = features[:, :, 0] - ( 
                slice_1.to(dtype).unsqueeze(1) * 0.16 - 69.04)
        return y1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_model = BugDemonstrator0().to(device)

features = torch.rand(12000, 100, 4).to(device)                                                                                                                                                                 
x = torch.rand(12000, 4).to(device)

inputs = [features, x]
input_names = ['x']
output_names = ['stack']

torch.onnx.export(pytorch_model, tuple(inputs), "test_voxelnet.onnx", verbose=True, input_names=input_names, output_names=output_names)