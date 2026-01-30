import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._inductor.compile_fx import compile_fx_inner

import torch.fx as fx
from torch._subclasses import FakeTensorMode

import onnx
from onnx import numpy_helper
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
model = model.to(device)
input_ids_1 = torch.tensor(
	[[tokenizer.encode("Here is some text to encode Hello World", add_special_tokens=True)]], device='cuda')

input_names = None
inputs_flatten = flatten(input_ids_1)
if input_names is None:
	input_names = []
	for i, _ in enumerate(inputs_flatten):
		input_names.append('input' + str(i+1))


opt_model = torch.compile(model, mode='max-autotune', fullgraph=True)
torch.onnx.export(opt_model, input_ids_1, "./", verbose=True, input_names=input_names,
			  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._inductor.compile_fx import compile_fx_inner

import torch.fx as fx
from torch._subclasses import FakeTensorMode

import onnx
from onnx import numpy_helper
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
model = model.to(device)
input_ids_1 = torch.tensor(
	[[tokenizer.encode("Here is some text to encode Hello World", add_special_tokens=True)]], device='cuda')

input_names = None
inputs_flatten = flatten(input_ids_1)
if input_names is None:
	input_names = []
	for i, _ in enumerate(inputs_flatten):
		input_names.append('input' + str(i+1))

torch.onnx.dynamo_export(model, input_ids_1, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)).save("gpt2.onnx")