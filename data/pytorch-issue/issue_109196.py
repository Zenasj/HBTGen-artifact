from transformers import AutoModelForSeq2SeqLM
import torch
import torch._dynamo
from torch._inductor.compile_fx import compile_fx

torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device('cuda:0')

with device:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-xl", torch_dtype=torch.float16
    ).eval()
    model.to(device)

compiled_forward = torch._dynamo.optimize(
    compile_fx,
    dynamic=True,
)(model.forward)

with torch.no_grad():
    toks = torch.ones(size=(12,1), dtype=int, device=device)
    compiled_forward(
        input_ids=toks,
        attention_mask=toks,
        decoder_input_ids=toks,
    )