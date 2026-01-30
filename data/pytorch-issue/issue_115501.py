import torch

export_program = torch.export.export(self._pytorch_nn_model, args=torch_input_args, kwargs=torch_inputs_kwargs)
onnx_program = torch.onnx.dynamo_export(export_program , *torch_input_args,
                                          **torch_input_kwargs, export_options=export_options)