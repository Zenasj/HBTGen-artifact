import torch

torch.onnx.export(model,
                  input_tuple,
                  'model.onnx',
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  do_constant_folding=False,
                  opset_version=12,
                  training=torch.onnx.TrainingMode.TRAINING,
                  dynamic_axes={'Input1' : {0 : 'batch_size'}, 'Input2' : {0 : 'batch_size'}})