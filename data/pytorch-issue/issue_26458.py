import torch.nn as nn

import onnxruntime
import torch
from torchvision import utils

if __name__ == "__main__":
    ort_session = onnxruntime.InferenceSession("stylegan.onnx")

    def to_numpy(tensor):
        return (tensor.detach() if tensor.requires_grad else tensor).cpu().numpy()

    z = torch.randn(1, 512)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(z)}
    ort_outs   = ort_session.run(None, ort_inputs)

    utils.save_image(
        torch.Tensor(ort_outs[0]),
        "onnx.png",
        nrow=1,
    )

import argparse
import math

import torch
from torch import onnx, nn
from torchvision import utils
import onnxruntime

from model import StyledGenerator

parser = argparse.ArgumentParser(description='ONNX')
parser.add_argument('-m', '--model', type=str, default='checkpoint/train_step-512.model')
parser.add_argument('-r', '--resolution', type=int, default=512)
args = parser.parse_args()


class Generator(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, z):

	# # if None is set
	# # ==============
	# # TypeError: i_(): incompatible function arguments. The following argument types are supported:
	# #    1. (self: torch._C.Node, arg0: str, arg1: int) -> torch._C.Node
    #     # Invoked with: %237 : Tensor = onnx::RandomNormal(), scope: Generator/StyledGenerator[generator]
    #     # , 'shape', 232 defined in (%232 : int[] = prim::ListConstruct(%228, %229, %230, %231), scope: Generator/StyledGenerator[generator]
    #     # ) (occurred when translating randn)

        noise = generate_noise(z.device) # None

        return self.generator(z, noise, step=7, alpha=1)


def generate_noise(device, step=7):
    noise = []
    for i in range(step + 1):
        resolution = 4 * 2 ** i
        noise.append(torch.randn(1, 1, resolution, resolution).to(device))
    return noise

if __name__ == "__main__":

    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StyledGenerator(args.resolution).to(device)

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['g_running'])

    model = Generator(model).to(device)
    model.eval()

    x_dummy = torch.randn(batch_size, 512, requires_grad=True).to(device)

    y_origin = model(x_dummy)
    utils.save_image(
        y_origin,
        f'sample/origin.png',
        nrow=math.ceil(math.sqrt(batch_size)),
        normalize=True,
        range=(-1, 1),
    )

    out = 'stylegan.onnx'
    input_names  = ['input_0']
    output_names = ['output_0']
    
    onnx.export(model, x_dummy, out, 
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes={
            'input_0': {0: 'batch_size'},
            'output_0': {0: 'batch_size'}
        }) 

    print("export has successfully completed.")