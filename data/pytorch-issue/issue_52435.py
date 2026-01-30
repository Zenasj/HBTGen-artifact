import torch
import torch.nn as nn
import torch.nn.functional as F

class MG(nn.Module):

    def __init__(self):
        super().__init__()
        # for test if torch.cat([bool, bool]) can convert

    def forward(self, x):
        kernel_preds = torch.randn([8, 4, 3, 3])
        preds = F.conv2d(x, kernel_preds,
                             stride=1)
        preds = preds.to(torch.float)
        preds = preds.sigmoid()
        seg_masks = preds > 0.03
        return seg_masks


torch_model = MG()
x = torch.randn([1, 4, 24, 24])
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,
                  "a.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,
                  verbose=True)
print('Done!')

class MG(nn.Module):

    def __init__(self):
        super().__init__()
        # for test if torch.cat([bool, bool]) can convert

    def forward(self, x, b):
        preds = F.conv2d(x, b,
                             stride=1)
        return preds


torch_model = MG()
x = torch.randn([1, 4, 24, 24])
b = torch.randn([8, 4, 3, 3])
torch_out = torch_model(x, b)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  (x, b),
                  "a.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,
                  verbose=True)
print('Done!')