import torch

def get_quantized_mlp():
    model = QuantizedMLP()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(0)
    for _ in range(32):
        model(get_mlp_input())
    torch.quantization.convert(model, inplace=True)
    return model