# torch.randint(0, 50257, (B, S), dtype=torch.long)  # GPT-2 vocab size is 50257
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import Module

class MyModel(Module):
    def __init__(self):
        super().__init__()
        # Initialize the model outside of any compiled region to avoid Dynamo tracing issues
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(torch.float)
    
    def forward(self, input_ids):
        # Use generate with fixed parameters matching the original issue's setup
        return self.model.generate(
            input_ids,
            max_length=4,
            do_sample=False,
            pad_token_id=self.model.config.eos_token_id,
        )

def my_model_function():
    # Returns initialized GPT-2 model as MyModel instance
    return MyModel()

def GetInput():
    # Create tokenizer and input tensor outside of compiled region
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer("Thanks for", return_tensors="pt").input_ids
    return input_ids

