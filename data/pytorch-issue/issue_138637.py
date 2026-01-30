import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/Phi-3.5-mini-instruct" # Slice error
model = AutoModelForCausalLM.from_pretrained(model_name)
# Export a 1 layer model otherwise onnx file is to big
model.model.layers = model.model.layers[:1]
model.config.num_hidden_layers = 1

dim = (1, 30)
input_ids = torch.randint(0, 32064, dim)  # Batch size 1, sequence length 30
attention_masks = torch.ones(*dim, dtype=torch.int64)

# Prepare the inputs for the model
inputs = {'input_ids': input_ids, 'attention_mask':attention_masks}

input_names = list(inputs.keys())
torch.onnx.export(model, inputs, "repro.onnx", input_names=input_names)

session = ort.InferenceSession(model_file, providers=ep, sess_opt=sess_opt)
outputs = session.run(None, inputs)