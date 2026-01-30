import torch.nn as nn

import torchvision
import torchaudio
import torch

# define a pytorch model
class SpecMaker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = torchvision.transforms.Compose(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=512,
                    win_length=512,
                    hop_length=256,
                ),
                torchaudio.transforms.AmplitudeToDB(top_db=100),
            ]
        )

    def forward(self, x):
        return self.transforms(x)


specmodel = SpecMaker()
input = torch.rand(32000 * 10)
spec = specmodel(input)
input_batch = torch.stack([input, input])
spec_batch = specmodel(input_batch) # just testing pytorch model works as expected

assert spec_batch.shape== torch.Size([2, 257, 1251])

onnx_program = torch.onnx.export(
    specmodel,
    (input_batch,),
    dynamic_shapes=[{0: "dim_x"}],
    report=True,
    dynamo=True,
)

onnx_program.save("specmodel2.onnx")

import onnx, onnxruntime
import torch

onnx_model = onnx.load("specmodel2.onnx")
onnx.checker.check_model(onnx_model)
input = torch.rand(32000 * 10)
input = torch.tensor((opso.birds).trim(0, 10).samples)

# what if its batched?
input_batched = torch.stack([input, input, input]) #works if batch has 2 samples, fails with 3 samples

EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]

ort_session = onnxruntime.InferenceSession("specmodel2.onnx", providers=EP_list)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_batched)}
ort_outs = ort_session.run(None, ort_inputs)

input = torch.tensor((opso.birds).trim(0, 10).samples)