import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

kwargs = dict(torch_dtype=torch.float32, use_cache=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", **kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = model.eval()

input_sentence =  "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"
inputs = tokenizer(input_sentence, return_tensors='pt')

with torch.inference_mode():
    for i in range(10):
        pre = time.time()
        output = model.generate(**inputs)
        print(f"eager eval time {i}: {time.time()-pre}")

model.generate = torch.compile(model.generate, backend='inductor', dynamic=True)
with torch.inference_mode():
    for i in range(10):
        pre = time.time()
        output_compile = model.generate(**inputs)
        print(f"compile eval time {i}: {time.time()-pre}")