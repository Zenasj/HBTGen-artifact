import torch

traced_script_module = torch.jit.trace(model, image)
traced_script_module.save(args.output_path)

torch.onnx.export(model, image,
onnx_model_save_path, opset_version=11, verbose=False, export_params=True, 
operator_export_type=OperatorExportTypes.ONNX,
input_names=['image'], output_names=['orient','conf','dim'])