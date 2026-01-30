import torch

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()

# quantization settings here for perchannel S8S8 quant
qconfig = torch.ao.quantization.qconfig.QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric, reduce_range=True),
    weight=default_per_channel_weight_observer
)

qconfig_mapping = QConfigMapping().set_global(qconfig)

model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, torch.randn(1,3,224,224))
calibrate(model_prepared, calib_data_loader)
model_quantized = quantize_fx.convert_fx(model_prepared)