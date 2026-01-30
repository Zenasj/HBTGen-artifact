py
import torch
import torch_tensorrt
import open_clip
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("convnext_xxlarge", 
                                           pretrained="laion2b_s34b_b82k_augreg_soup")

    model = model.visual.eval().to(device)
    image_size = model.image_size
    image = torch.randn((1, 3, *image_size)).to(device) # (1, 3, 256, 256)
    model = torch_tensorrt.compile(model, ir="dynamo", inputs=[image])

    trt_ep = torch.export.export(model, (image,))
    torch.export.save(trt_ep, "convnext_xxlarge_compiled.ep")