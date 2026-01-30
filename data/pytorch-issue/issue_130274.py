import torch
import torch.export._trace
import transformers
from torch import bfloat16
from transformers import AutoTokenizer
from torch_xla.stablehlo import exported_program_to_stablehlo
import argparse


def main(model_name):
    # Load tokenizer and model
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    name = model_name.split("/")[-1]
    cache_dir = "/models/"
    model = model_name
    # Load tokenizer and model using the specified cache directory
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()

    base_prompt = "How many hours are in a day?"
    base_inputs = tokenizer(base_prompt, return_tensors="pt")
    input_ids = base_inputs.input_ids

    print("Exporting model using torch export...")
    exported = torch.export.export(
        model,
        (input_ids,),
    )