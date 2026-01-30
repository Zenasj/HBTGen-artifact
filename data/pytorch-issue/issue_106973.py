import onnx
import torch

from conv_tasnet import ConvTasNet


def convertoOnnx():
    device = torch.device('cpu')
    # Create model.
    model = ConvTasNet(256, 20, 256, 512, 3, 8, 4,
                       2, norm_type="gLN", causal=0,
                       mask_nonlinear="relu")

    model.to(device)
    dummy_input =  {'mixtures': torch.ones(256, 20).to(torch.device('cpu'))}

    onnx_model_path = 'conv_tasnet.onnx'
    torch.onnx.export(model, dummy_input["mixtures"], onnx_model_path, verbose=True, opset_version=12)
def main():
    convertoOnnx()

if __name__ == "__main__":

    main()