import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.backend_config import get_onednn_backend_config

qengine = 'onednn'
torch.backends.quantized.engine = qengine
qconfig_mapping = get_default_qconfig_mapping(qengine)
prepared_model = prepare_fx(model_fp32, qconfig_mapping, \
                            example_inputs=x, backend_config=get_onednn_backend_config())
quantized_model = convert_fx(prepared_model, backend_config=get_onednn_backend_config())