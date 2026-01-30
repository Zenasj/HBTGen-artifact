import torch.nn as nn

import torch
import onnx

def test():

    class SumInt32(torch.nn.Module):

        def forward(self, a):
            return torch.sum(a, dtype=torch.int32)
        
    sumi = SumInt32().eval()
    assert sumi(torch.randint(42, (17,),)).dtype == torch.int32 
    
    torch.onnx.export(sumi, (torch.randint(42, (17,),)), "/tmp/sumi.onnx" )
    model = onnx.load("/tmp/sumi.onnx")
    
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.INT32

test()

assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
AssertionError

def test(byte_data):
  import onnx

  model = onnx.ModelProto()
  model.ParseFromString(byte_data)

  assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
  print("Passed...")

# data type = int 32
byte_data = b'\x08\x07\x12\x07pytorch\x1a\x052.0.0:\xe9\x01\n6\n\x0connx::Cast_0\x12\x0e/Cast_output_0\x1a\x05/Cast"\x04Cast*\t\n\x02to\x18\x06\xa0\x01\x02\n<\n\x0e/Cast_output_0\x12\x10/Cast_1_output_0\x1a\x07/Cast_1"\x04Cast*\t\n\x02to\x18\x07\xa0\x01\x02\n=\n\x10/Cast_1_output_0\x12\x013\x1a\n/ReduceSum"\tReduceSum*\x0f\n\x08keepdims\x18\x00\xa0\x01\x02\x12\ttorch_jitZ\x1a\n\x0connx::Cast_0\x12\n\n\x08\x08\x07\x12\x04\n\x02\x08\x11b\x0b\n\x013\x12\x06\n\x04\x08\x06\x12\x00B\x02\x10\x0e'
test(byte_data)

# data type = int 64
byte_data = b'\x08\x07\x12\x07pytorch\x1a\x052.0.0:\xe9\x01\n6\n\x0connx::Cast_0\x12\x0e/Cast_output_0\x1a\x05/Cast"\x04Cast*\t\n\x02to\x18\x06\xa0\x01\x02\n<\n\x0e/Cast_output_0\x12\x10/Cast_1_output_0\x1a\x07/Cast_1"\x04Cast*\t\n\x02to\x18\x07\xa0\x01\x02\n=\n\x10/Cast_1_output_0\x12\x013\x1a\n/ReduceSum"\tReduceSum*\x0f\n\x08keepdims\x18\x00\xa0\x01\x02\x12\ttorch_jitZ\x1a\n\x0connx::Cast_0\x12\n\n\x08\x08\x07\x12\x04\n\x02\x08\x11b\x0b\n\x013\x12\x06\n\x04\x08\x07\x12\x00B\x02\x10\x0e'
test(byte_data)