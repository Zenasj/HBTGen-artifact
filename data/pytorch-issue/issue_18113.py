import torch.nn as nn
import torch
import torch.nn.functional as F

class Test(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        #return F.upsample(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)
                                # RuntimeError: ONNX symbolic expected a constant value in the trace

        #return F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)
                                # RuntimeError: ONNX symbolic expected a constant value in the trace

        #return F.upsample(x, size=(600, 600), mode='bilinear', align_corners=False)
                                # UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.

        #return F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=True)
                                # UserWarning: ONNX export failed on upsample_bilinear2d because align_corners == True not supported
                                # RuntimeError: ONNX export failed: Couldn't export operator aten::upsample_bilinear2d

        return F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=False) #no warning, all clear

model = Test()
x = torch.zeros((1, 3, 300, 300))
torch.onnx._export(model, x, "test.onnx", verbose=True)

3
import onnx
from onnx import version_converter, helper

# load model
original_model  = onnx.load(model_path)

# converts oppset v9 to v8
converted_model = version_converter.convert_version(original_model, 8)

# change attribute of all Upsample nodes
for node in converted_model.graph.node:
    if node.op_type == 'Upsample':
        # get id-attribute_name map
        id = { attribute.name: id for id, attribute in enumerate(node.attribute)}
        # get & remove "scales" attribute
        att_scales = node.attribute.pop(id['scales']) 
        _, _, scale_height, scale_width = att_scales.floats # CARE IT DEPENDS ON ORDER. HERE [B, C, W, H] IS EXPECTED
        # append new attributes 'scale_width' & 'scale_height'
        node.attribute.extend([
            helper.make_attribute('width_scale', scale_width),
            helper.make_attribute('height_scale', scale_height)
        ])

# save
onnx.save(converted_model, result_path)