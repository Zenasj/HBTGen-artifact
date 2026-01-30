import torch
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modelpath=r"Z:\AI_SDK\CPP_GFPGAN\Pretrained_MODELS_GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth"
channel_multiplier=2
model = GFPGANv1Clean(
    out_size=512,
    num_style_feat=512,
    channel_multiplier=channel_multiplier,
    decoder_load_path=None,
    fix_decoder=False,
    num_mlp=8,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True)

w=torch.load(modelpath)
model.load_state_dict(w,strict=False)
model.to(device)
model.eval()
inputs = torch.ones((1, 3, 512,512)).to(device)
onnxpath=r"Z:\AI_SDK\CPP_GFPGAN\Pretrained_MODELS_GFPGAN\experiments\pretrained_models\GFPGANv1.3.onnx"
print(model)
torch.onnx.export(model,inputs,onnxpath,export_params=True,verbose=True,input_names=['input'],output_names=['output'],opset_version=12)