import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization import prepare_fx

from torch.ao.quantization.quantize_fx import prepare_fx