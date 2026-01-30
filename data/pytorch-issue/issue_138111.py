import torch

from models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 384
image = torch.randn(1, 3,384,384)
caption_input = ""

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(image,caption_input,), strict=False)