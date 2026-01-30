import torch.nn as nn

m = nn.Sequential(nn.Linear(1, 1), nn.ReLU()).eval()
qconfig_dict = { 
    'object_type': [ 
        (nn.Linear, default_dynamic_qconfig),
    ],
} 
mp = quantize_fx.prepare_fx(m)
mq = quantize_fx.convert_fx(mp)