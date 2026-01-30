import torch    
import torch.onnx    
import torch.nn as nn    
import torch.nn.functional as F    
    
DIM, TRANSPOSE, FUSE = 1, False, False   # quantized::conv1d                  (fails to export)    
#DIM, TRANSPOSE, FUSE = 1, False, True   # quantized::conv1d_relu             (supported)    
#DIM, TRANSPOSE, FUSE = 2, False, False  # quantized::conv2d                  (supported)    
#DIM, TRANSPOSE, FUSE = 2, False, True   # quantized::conv2d_relu             (supported)    
#DIM, TRANSPOSE, FUSE = 3, False, False  # quantized::conv3d                  (fails to export)    
#DIM, TRANSPOSE, FUSE = 3, False, True   # quantized::conv3d_relu             (fails to export)    
#DIM, TRANSPOSE, FUSE = 1, True, False   # quantized::conv_transpose1d        (fails to export)    
#DIM, TRANSPOSE, FUSE = 1, True, True    # quantized::conv_transpose1d_relu   (no fuser)    
#DIM, TRANSPOSE, FUSE = 2, True, False   # quantized::conv_transpose2d        (fails to export)    
#DIM, TRANSPOSE, FUSE = 2, True, True    # quantized::conv_transpose2d_relu   (no fuser)    
#DIM, TRANSPOSE, FUSE = 3, True, False   # quantized::conv_transpose3d        (fails to export)    
#DIM, TRANSPOSE, FUSE = 3, True, True    # quantized::conv_transpose3d_relu   (no fuser)    
    
    
if TRANSPOSE:    
    CONV_OP = getattr(nn, f"ConvTranspose{DIM}d")    
else:    
    CONV_OP = getattr(nn, f"Conv{DIM}d")    
    
class Model(nn.Module):    
    def __init__(self):    
        super().__init__()    
        self.conv = CONV_OP(3, 3, 3)    
        self.relu = nn.ReLU()    
        self.quant = torch.ao.quantization.QuantStub()    
        self.dequant = torch.ao.quantization.DeQuantStub()    
    
    def forward(self, x):    
        z = x    
        z = self.quant(z)    
        z = self.conv(z)    
        z = self.relu(z)    
        z = self.dequant(z)    
        return z    
    
    
model = Model()    
model.eval()    
    
input_shape = (1, 3) + (32,) * DIM    
    
model_fp32 = model    
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")    
if FUSE:    
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [["conv", "relu"]])    
else:    
    model_fp32_fused = model_fp32    
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)    
input_fp32 = torch.randn(*input_shape)    
# PTQ    
model_fp32_prepared(input_fp32)    
    
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)    
model = model_int8    
    
torch.onnx.export(    
    model,    
    input_fp32,    
    "output.onnx",    
    opset_version=16,       
    input_names=["x"],    
    output_names=["y"],    
)