import torch

inputs = image.unsqueeze(0)
net.to(torch.device('cpu'))
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(72) ]
output_names = [ "output1", "output2"]
torch.onnx.export(net,      # model being run
        inputs,                         # model input (or a tuple for multiple inputs)
        "sbr.onnx",                # where to save the model (can be a file or file-like object)
        verbose=True, input_names=input_names, output_names=output_names
       ,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        ,opset_version=12                  
        )