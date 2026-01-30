import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from train import load_model

hparams = create_hparams()
hparams.sampling_rate = 22050
checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
_ = model.eval()

text_inputs = torch.randn(64, 163, device='cpu')
text_legths = torch.randn(64, device='cpu')
mels = torch.randn(64, 80, 851, device='cpu')
max_len = 163
output_lengths = torch.randn(64, device='cpu')
dummy_input = (text_inputs, text_legths, mels, max_len, output_lengths)

torch.onnx.export(model, dummy_input, "tacotron.onnx", verbose=True)

torch.onnx.export(
        model, 
        torch.zeros(batch_size, img_channel, *img_size).shape,
        path, 
        opset_version=12,
        verbose=False,
        input_names=['images'],
        output_names=['output'],
    )

torch.onnx.export(
        model, 
        torch.zeros(batch_size, img_channel, *img_size),
        path, 
        opset_version=12,
        verbose=False,
        input_names=['images'],
        output_names=['output'],
    )