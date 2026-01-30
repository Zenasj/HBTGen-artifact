import torch

conf_1 = torch.ao.quantization.backend_config.backend_config.DTypeConfig()
print(conf_1)

conf_2 = torch.ao.quantization.backend_config.backend_config.BackendConfig()
print(conf_2)

conf_3 = torch.ao.quantization.backend_config.backend_config.BackendPatternConfig()
print(conf_3)

conf_4 = torch.ao.quantization.fx.custom_config.PrepareCustomConfig()\
    .set_input_quantized_indexes([0])
print(conf_4)

conf_5 = torch.ao.quantization.fx.custom_config.ConvertCustomConfig()\
    .set_preserved_attributes(['foo'])
print(conf_5)

conf_6 = torch.ao.quantization.fx.custom_config.FuseCustomConfig()\
    .set_preserved_attributes(['foo'])
print(conf_6)