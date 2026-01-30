from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.onnx

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = model.cpu() # Trying to do it while on the GPU results in an error of tensors being on different devices even after moving the inputs to cuda.
inputs = tokenizer("What is your name?", return_tensors="pt").input_ids
torch.onnx.dynamo_export(model, inputs).save('llama-2-7b.onnx')

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
import torch.onnx

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
gptq_config = GPTQConfig(bits=8, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config = gptq_config)
model = model.cpu() # Trying to do it while on the GPU results in an error of tensors being on different devices even after moving the inputs to cuda.
inputs = tokenizer("What is your name?", return_tensors="pt").input_ids
torch.onnx.dynamo_export(model, inputs).save('llama-2-7b.onnx')

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
import torch.onnx
from peft import get_peft_model
from peft import LoraConfig


tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
gptq_config = GPTQConfig(bits=8, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config = gptq_config)
lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=['q_proj','k_proj','v_proj','up_proj','down_proj'],
            bias="none",
            task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model = model.cpu() # Trying to do it while on the GPU results in an error of tensors being on different devices even after moving the inputs to cuda.
inputs = tokenizer("What is your name?", return_tensors="pt").input_ids
torch.onnx.dynamo_export(model, inputs).save('llama-2-7b.onnx')