import torch
import torch.nn as nn
import numpy as np

class FixedUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest', align_corners=False):
        super(FixedUpsample, self).__init__()
        self.mode=mode
        self.align=align_corners
        self.scale=scale_factor
    def forward(self,x):
        _,_,h,w = x.shape
        return nn.Upsample(size=(h*self.scale,w*self.scale),mode=self.mode,)(x)

torch.onnx.export(FixedUpsample(mode='bilinear'),
                  img,
                  'test_fixedUpsample.onnx', 
                  verbose=False, 
                  export_params=True, 
                  keep_initializers_as_inputs=True,
                  do_constant_folding=True,
                  opset_version=12, 
                  operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
onnx.checker.check_model(onnx.load('test_fixedUpsample.onnx'))

import onnxruntime as ort
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
ort_sess = ort.InferenceSession(str(onnx_path))
ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_sess.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)