import torch
print(torch.__version__)
# '1.4.0'

# check.py
import sys
import onnx
onnx_model = onnx.load(sys.argv[1])
print("Producer Name:", onnx_model.producer_name)
print("Producer Version:", onnx_model.producer_version)
print("Opset", onnx_model.opset_import[0])