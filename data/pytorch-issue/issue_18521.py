import torch.nn as nn

import torch, onnx 
import caffe2.python.onnx.backend as onnx_caffe2_backend      
class Model(torch.nn.Module): 
  def __init_(self): 
    super().__init__() 
  def forward(self,x): 
    return torch.stft(x, 320, 160, window=torch.hann_window(320)) 
m = Model()
x = torch.ones(1,1,512)
x2 = x.reshape(1,-1)
# 0 
torch.onnx.export(m, x2, f="specmodel.onnx", verbose=True,
  operator_export_type=torch.onnx.OperatorExportTypes.ONNX) 

# 1
torch.onnx.export(m, x2, f="specmodel.onnx", verbose=True,
  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN) 
model = onnx.load('specmodel.onnx')                                                                                                                                             
prepared_backend = onnx_caffe2_backend.prepare(model)                                                                                                                           
W = {model.graph.input[0].name: x2.data.numpy()}                                                                                                                                
c2_out = prepared_backend.run(W)[0]    

# 2 
torch.onnx.export(m, x2, f="specmodel.onnx", verbose=True,
  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK) 
model = onnx.load('specmodel.onnx')                                                                                                                                             
prepared_backend = onnx_caffe2_backend.prepare(model)                                                                                                                           
W = {model.graph.input[0].name: x2.data.numpy()}                                                                                                                                
c2_out = prepared_backend.run(W)[0]