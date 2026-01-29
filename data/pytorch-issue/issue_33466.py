# torch.rand(1, 4, 4, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(2, 4, 2, 2))  # Matches original weight shape
        self.strides = (1, 1)
        self.pads = (0, 0)
        self.dilations = (1, 1)
        self.groups = 1

    def forward(self, x):
        # Quantize input and weights with scales from the issue
        qx = torch.quantize_per_tensor(x, scale=0.052, zero_point=0, dtype=torch.quint8)
        qweight = torch.quantize_per_tensor(self.weight, scale=2.39, zero_point=0, dtype=torch.qint8)
        
        orig_engine = torch.backends.quantized.engine
        result = torch.tensor([False], dtype=torch.bool)
        
        try:
            # Run FBGEMM path
            torch.backends.quantized.engine = 'fbgemm'
            w_prepack_fbgemm = torch.ops.quantized.conv2d_prepack(qweight, None, self.strides, self.pads, self.dilations, self.groups)
            out_fbgemm = torch.ops.quantized.conv2d(qx, w_prepack_fbgemm, self.strides, self.pads, self.dilations, self.groups, 0.112, 0)
            
            # Run QNNPACK path
            torch.backends.quantized.engine = 'qnnpack'
            w_prepack_qnnpack = torch.ops.quantized.conv2d_prepack(qweight, None, self.strides, self.pads, self.dilations, self.groups)
            out_qnnpack = torch.ops.quantized.conv2d(qx, w_prepack_qnnpack, self.strides, self.pads, self.dilations, self.groups, 0.112, 0)
            
            # Compare dequantized outputs
            deq_fbgemm = out_fbgemm.dequantize()
            deq_qnnpack = out_qnnpack.dequantize()
            result = torch.tensor([torch.allclose(deq_fbgemm, deq_qnnpack)], dtype=torch.bool)
        except Exception:
            # QNNPACK error indicates mismatch
            result = torch.tensor([False], dtype=torch.bool)
        finally:
            torch.backends.quantized.engine = orig_engine  # Restore original backend
            
        return result  # Returns True if outputs match (or QNNPACK succeeded)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 4, 4, 4)

