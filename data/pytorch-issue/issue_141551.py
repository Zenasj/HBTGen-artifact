import torch
from torchaudio.models import Conformer

input_dim=80
model = Conformer(input_dim,
                  num_heads=4,
                  ffn_dim=128,
                  num_layers=4,
                  depthwise_conv_kernel_size=31,
 )
lengths = torch.randint(1, 400, (10,)) 
input_data = torch.rand(10, int(lengths.max()), input_dim) 
model.eval()
exported_program = torch.export.export(model, (input_data, lengths))