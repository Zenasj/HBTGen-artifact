import torch

torch.onnx.export(quantized_model,
                  (inputs[0], inputs[1]),
                  'quantized_model.onnx',
                  opset_version=13,
                  do_constant_folding=True,  
                  input_names=["batch_input_ids", "batch_att_mask"],  
                  output_names=["logits"], 
                  dynamic_axes={"batch_input_ids": [0], "batch_att_mask": [0],
                                "logits": [0]})