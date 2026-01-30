from pathlib import Path
import torch
import torchvision

def main():
    weights_name = "SSD300_VGG16_Weights.DEFAULT"
    # Get the pretrained ssd300_vgg16 model from torchvision.models
    model = torchvision.models.get_model("ssd300_vgg16", weights=weights_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 480, 480)
    ROOT = Path(__file__).parent.resolve()
    fp32_onnx_path = f"{ROOT}/ssd300_vgg16_fp32.onnx"
    torch.onnx.export(model.cpu(), dummy_input, fp32_onnx_path)


if __name__ == "__main__":
    main()

from pathlib import Path
import torch
import torchvision

def main():
    weights_name = "SSD300_VGG16_Weights.DEFAULT"
    # Get the pretrained ssd300_vgg16 model from torchvision.models
    model = torchvision.models.get_model("ssd300_vgg16", weights=weights_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 480, 480)
    ROOT = Path(__file__).parent.resolve()
    fp32_onnx_path = f"{ROOT}/ssd300_vgg16_fp32.onnx"
    onnx_program = torch.onnx.dynamo_export(model.cpu(), dummy_input)
    onnx_program.save(fp32_onnx_path)


if __name__ == "__main__":
    main()