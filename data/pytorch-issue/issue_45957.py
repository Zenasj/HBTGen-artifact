import torch

torch.onnx.export(onnx_model,                # model being run
                  inputs,                    # model input (or a tuple for multiple inputs)
                  "en_v1_test.onnx",         # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0: 'batch',
                                           1: 'samples'},    
                                'output' : {0: 'batch',
                                            1: 'frames'}},
                  verbose=True
                 )