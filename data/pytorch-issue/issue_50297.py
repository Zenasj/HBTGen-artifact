import torch.nn as nn

my_mapping = get_default_static_quant_module_mappings()
my_mapping[nn.Linear] = UserLinearImplementation
model_A = convert(model_A, mapping=my_mapping)

default_mapping = get_default_static_quant_module_mappings()
model_B = convert(model_B, mapping=default_mapping)