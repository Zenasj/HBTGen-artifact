import torch
import torch.nn as nn

# creating a qconfig
observer_class_wt = torch.quantization.PerChannelMinMaxObserver
observer_class_act = torch.quantization.HistogramObserver # very low gpu utilization as reported by nvidia-smi
# observer_class_act = torch.quantization.MovingAverageMinMaxObserver # If I use this, I get much higher GPU utilization.
qscheme_wt = torch.per_channel_affine
qscheme_act = torch.per_tensor_affine
reduce_range_wt=False
reduce_range_act=False
kwargs_wt = {
    'reduce_range':reduce_range_wt
}
kwargs_act = {
    'reduce_range':reduce_range_act
}
kwargs_wt['quant_min'], kwargs_wt['quant_max'] = (-64, 63) if kwargs_wt['reduce_range'] else (-128, 127)
kwargs_act['quant_min'], kwargs_act['quant_max'] = (0, 127) if kwargs_act['reduce_range'] else (0, 255)
kwargs_wt['dtype'] = torch.qint8
kwargs_act['dtype'] = torch.quint8

weight_class = torch.quantization.fake_quantize.FakeQuantize.with_args(observer=observer_class_wt, **kwargs_wt)
activation_class = torch.quantization.fake_quantize.FakeQuantize.with_args(observer=observer_class_act, **kwargs_act)
myQconfig = torch.quantization.qconfig.QConfig(activation=activation_class, weight=weight_class)

qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
# qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
qat_model.qconfig = myQconfig


torch.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)
qat_model.cuda()
qat_model = nn.DataParallel(qat_model)