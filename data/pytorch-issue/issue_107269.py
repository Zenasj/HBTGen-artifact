import torch.nn as nn

import torch
import sys
import random
import numpy as np

# from stop_word import StoppingCriteriaSub, StoppingCriteriaList
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

device = "cuda:0"
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-30b",legacy=False)
model = LlamaForCausalLM.from_pretrained(
        "huggyllama/llama-30b",
        quantization_config=bnb_config,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model) ### seems here is the bug
tokens = tokenizer(["prompt is prompt is prompt","her eyes are so beautiful"], return_tensors='pt', padding=True).to(device)
with torch.no_grad():
    logits = model(**tokens, return_dict=True).logits