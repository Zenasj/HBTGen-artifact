import torch
import segmentation_models_pytorch as smp
import torch.ao.quantization as tq

model = smp.Unet(
    encoder_name='mobilenet_v2',       
    in_channels=3,                  
    classes=1,               
)

model = tq.QuantWrapper(model)
model.qconfig = tq.get_default_qat_qconfig('fbgemm')
q_model = tq.prepare_qat(model)
int8_model = tq.convert(q_model)

image = image.to('cpu')
logits_mask = int8_model(image)
prob_mask = logits_mask.sigmoid()
pred_mask = (prob_mask > 0.5).float()