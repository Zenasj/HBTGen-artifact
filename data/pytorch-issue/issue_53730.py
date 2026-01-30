import torch

_ = torch.onnx.export(decoder_with_lm_head_init,
                            (input_ids_dec, attention_mask, enc_out),
                            f"{output_prefix}-decoder-with-lm-head_initial.onnx",
                            export_params=True,
                            opset_version=12,
                            input_names=['input_ids', 'encoder_attention_mask', 'encoder_hidden_states'],
                            output_names=['logits', 'past_key_values'],
                            dynamic_axes={
                              'input_ids': {0:'batch', 1: 'sequence'}, # batch_size, seq_length = input_shape
                              'encoder_hidden_states': {0:'batch', 1: 'sequence'},
                              'logits': {0:'batch', 1: 'sequence'},
                              "past_key_values" : {0:'batch', 1: 'sequence'},
                              'encoder_attention_mask' : {0:'batch', 1: 'sequence'},
                            } 
                            )