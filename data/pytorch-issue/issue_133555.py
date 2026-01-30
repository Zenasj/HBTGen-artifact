import torch
import torch.onnx

import torchvision

torch_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2( weights='DEFAULT') # doesn't work
#torch_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()  # doesn't work
#torch_model = torchvision.models.segmentation.deeplabv3_resnet50() # works fine!
torch_model.eval()
torch_input = torch.randn(1, 3, 32, 32)

is_dynamo_export = True
if (is_dynamo_export):
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_program.save("onnx_dynamo_export_ResNET50.onnx")        

is_common_export = True
if (is_common_export):
    torch.onnx.export(torch_model,               # model being run
                      torch_input,                         # model input (or a tuple for multiple inputs)
                      "onnx_export_ResNET50.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

torch_model = torchvision.models.segmentation.deeplabv3_resnet50()

torch_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()

opset_version=17,          # the ONNX version to export the model to