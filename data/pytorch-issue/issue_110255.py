import torch.nn as nn
import random

import numpy as np
import torch

model = torch.nn.Transformer(d_model=16,
                            nhead=4,
                            num_encoder_layers=1,
                            num_decoder_layers=1,
                            batch_first=False)

src = torch.tensor(np.random.randn(3,1,16).astype(np.float32))
tgt = torch.tensor(np.random.randn(5,1,16).astype(np.float32))

torch.onnx.export(model,
                  {'src':src, 'tgt':tgt},
                  'torch_nn_transformer.onnx',
                  verbose=False,
                  input_names=['src', 'tgt'],
                  opset_version=15,
                  output_names=['out'],
                  dynamic_axes={
                    'src': {0: 'src_len'},
                    'tgt': {0: 'tgt_len'},
                    'out': {0: 'tgt_len'}
                  })