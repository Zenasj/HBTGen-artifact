import math

import torch
from deepvoice3_pytorch.tests.test_deepvoice3 import _test_data
from deepvoice3_pytorch.train import build_model
from deepvoice3_pytorch.train import restore_parts, load_checkpoint

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = build_model()
model = load_checkpoint(checkpoint_path, model, None, True)

sequence = np.array(synthesis._frontend.text_to_sequence(text, p=0))
sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)

dummy_model_input = (sequence, text_positions)

output_onnx_model ='deepvoice3.onnx'

torch.onnx.export(model, 
                  dummy_model_input, 
                  output_onnx_model,
                  input_names=['sequence', 'text_positions'], 
                  dynamic_axes={'sequence': {1:'sequence'}, 'text_positions': {1: 'pos'}},
                  output_names=['mel_outputs', 'linear_outputs', 'alignments', 'done'],
                  opset_version=11,
                  verbose=True)

device = torch.device("cpu")

model = build_model()
model = load_checkpoint(checkpoint_path, model, None, True)
model.eval()

import torch
#from tests.test_deepvoice3 import _test_data
from train import build_model
from train import restore_parts, load_checkpoint
import train
import synthesis
import numpy as np
from deepvoice3_pytorch import frontend
synthesis._frontend = getattr(frontend, "en")
train._frontend =  getattr(frontend, "en")

checkpoint_path = "/home/ksenija/Downloads/20180505_deepvoice3_checkpoint_step000640000.pth"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = build_model()
model = load_checkpoint(checkpoint_path, model, None, True)

text = "test"
sequence = np.array(synthesis._frontend.text_to_sequence(text, p=0))
sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)

dummy_model_input = (sequence, text_positions)

output_onnx_model ='deepvoice3.onnx'

torch.onnx.export(model, 
                  dummy_model_input, 
                  output_onnx_model,
                  input_names=['sequence', 'text_positions'], 
                  dynamic_axes={'sequence': {1:'sequence'}, 'text_positions': {1: 'pos'}},
                  output_names=['mel_outputs', 'linear_outputs', 'alignments', 'done'],
                  opset_version=11,
                  verbose=True)

from deepvoice3_pytorch.train import build_model
from deepvoice3_pytorch.train import restore_parts, load_checkpoint

device = torch.device("cpu")

model = build_model()
model = load_checkpoint(checkpoint_path, model, None, True)

text = "Hello World" 
sequence = np.array(synthesis._frontend.text_to_sequence(text, p=0))
sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)

mel_outputs, linear_outputs, alignments, done = model(sequence, text_positions=text_positions)

@torch.jit.script
def attention_script(x, values):
    s = int(values.size(1))
    div = int(1.0) / s
    t = math.sqrt(div)
    x = x * (s * t)
    return x

import torch
from train import build_model
from train import restore_parts, load_checkpoint
import train
import synthesis
import numpy as np
import hparams
import json
from deepvoice3_pytorch import frontend
synthesis._frontend = getattr(frontend, "en")
train._frontend =  getattr(frontend, "en")

preset = "/home/ksenija/Downloads/20180505_deepvoice3_ljspeech.json"
checkpoint_path = "/home/ksenija/Downloads/20180505_deepvoice3_checkpoint_step000640000.pth"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load parameters from preset
with open(preset) as f:
  hparams.hparams.parse_json(f.read())

model = build_model()
model = load_checkpoint(checkpoint_path, model, None, True)
model.to(device)
model.eval()

text = "Generative adversarial network or variational auto-encoder."

def helper(model, text, p=0):
  sequence = np.array(synthesis._frontend.text_to_sequence(text, p=p))
  sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
  text_positions = torch.randn(sequence.size(0), 12, 80)
 
  return model, sequence, text_positions

model, sequence, text_positions = helper(model, text)

def convert_to_onnx(model, sequence, text_positions):
  dummy_model_input = (sequence, text_positions)
  output_onnx_model ='deepvoice3.onnx'

  torch.onnx.export(model, 
                    dummy_model_input, 
                    output_onnx_model,
                    input_names=['sequence', 'text_positions'], 
                    dynamic_axes={'sequence': {0:'sequence'}, 'text_positions': {1: 'pos'}},
                    output_names=['mel_outputs', 'linear_outputs', 'alignments', 'done'],
                    opset_version=12,
                    verbose=True)

convert_to_onnx(model, sequence, text_positions)

import onnxruntime

ort_session = onnxruntime.InferenceSession('deepvoice3.onnx')

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(sequence), ort_session.get_inputs()[1].name: to_numpy(text_positions)}
ort_outs = ort_session.run(None, ort_inputs)
torch_outs = model(sequence, text_positions)

[np.testing.assert_allclose(to_numpy(out), ort_out, rtol=1e-04, atol=1e-04) for out, ort_out in zip(torch_outs, ort_outs)]

text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)

text_positions = torch.randn(sequence.size(0), 12, 80)

ort_outs