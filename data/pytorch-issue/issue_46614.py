import torch
import torch.nn as nn

model = nn.Sequential(
                nn.Linear(5, 5),
                nn.Linear(5, 5),
            ).eval()
qengine = torch.backends.quantized.engine
qconfig_dict = {'': torch.quantization.get_default_qconfig(qengine)}

 # symbolically trace
model = symbolic_trace(model)
model = prepare_fx(model, qconfig_dict)