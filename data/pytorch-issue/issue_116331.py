import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run(architecture: str, context_length: int, batch_size) -> None:
    model = AutoModelForCausalLM.from_pretrained(architecture)
    model = model.eval().to("mps")

    input_ids = torch.zeros((batch_size, context_length), dtype=torch.int64)
    input_ids = input_ids.to(model.device)

    with torch.inference_mode():
        output = model(input_ids)

    print(f"Model predictions for first few tokens for {input_ids.shape=}:")
    for pos in range(3):
        token_id = torch.argmax(output.logits[0, pos]).item()
        logit = output.logits[0, pos, token_id].item()
        print(f" {pos}: {token_id=:>4} ({logit=:>9.2f})")

# BASELINE
run("gpt2-medium", context_length=1023, batch_size=1)  
# Model predictions for first few tokens for input_ids.shape=torch.Size([1, 1023]):
# 0: token_id= 198 (logit=   -46.95)
# 1: token_id= 198 (logit=   -43.56)
# 2: token_id=   0 (logit=   -20.67)

# CORRECT PREDICTIONS (SAME LIKE BEFORE)
run("gpt2-medium", context_length=1024, batch_size=2)
# Model predictions for first few tokens for input_ids.shape=torch.Size([2, 1024]):
#  0: token_id= 198 (logit=   -46.95)
#  1: token_id= 198 (logit=   -43.56)
#  2: token_id=   0 (logit=   -20.67)

# WRONG PREDICTIONS (DIFFERENT THAN BEFORE)
run("gpt2-medium", context_length=1024, batch_size=1)
# Model predictions for first few tokens for input_ids.shape=torch.Size([1, 1024]):
# 0: token_id=16706 (logit=   -25.97)
# 1: token_id= 198 (logit=   -46.95)
# 2: token_id= 198 (logit=   -51.52)

3
# Replace this
# x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
# with:
x = x.view(-1, x.size(-1)) @ self.weight + self.bias

import torch

def do_mm(n:int, device:str):
    x=torch.eye(n, device=device)
    bias=torch.arange(n, dtype=torch.float32, device=device)
    print(torch.addmm(bias, x, x), "addmm")
    print(x @ x + bias, "mm+add")

if __name__ == "__main__":
    do_mm(2, "mps")
    do_mm(2, "cpu")