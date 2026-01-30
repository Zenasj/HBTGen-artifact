import torch.nn as nn

import torch
import time
from glow import WaveGlow
model = "waveglow_276000"

wg = torch.load(model, map_location=torch.device("cpu"))["model"]
wg.eval()
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    wg, {torch.nn.ConvTranspose1d, torch.nn.Conv1d}, dtype=torch.qint8
)

import torch
import time
from glow import WaveGlow
model = "waveglow_276000"

wg = torch.load(model, map_location=torch.device("cpu"))["model"]
wg.eval()
ip = torch.randn(1, 80, 200)
op = wg.infer(ip)

import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    wg, {torch.nn.ConvTranspose1d, torch.nn.Conv1d}, dtype=torch.qint8
)