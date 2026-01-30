import torch
import transformers
from transformers import LlamaConfig

kwargs = {}
device = "cpu"
max_length = 512
batch_size = 1
config = LlamaConfig(num_hidden_layers=16)
model = transformers.AutoModelForCausalLM.from_config(config, **kwargs).to(device)

eval_context = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
example_inputs = {'input_ids': eval_context, }
model.eval()

module = torch.jit.trace(model, example_kwarg_inputs=example_inputs)