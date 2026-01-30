def convert_to_onnx(features):
    import torch.onnx
    torch.onnx.export(model,
                      features,
                      '/home/rajeev/pb/espnet/agent_model/onnx_model/model_aten.onnx',
                      input_names= ['features'],
                      output_names=['hyp_text'],
                      dynamic_axes={
                          'features':{0:'batch_size',1: 'time_steps'},
                          'hyp_text':{0:'batch_size'}
                      },
                      verbose=True,
                      opset_version=12,
                      # operator_export_type= torch.onnx.OperatorExportTypes.ONNX_ATEN,
                      use_external_data_format=True,
                      enable_onnx_checker=True,
                      do_constant_folding=True
                      )

try:
    onnx_options = SessionOptions()
    _ = InferenceSession(path, onnx_options)
    print("Model correctly loaded")
except RuntimeException as re:
    print("Error while loading the model: {}".format(re))