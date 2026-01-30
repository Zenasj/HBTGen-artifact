from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch._dynamo as torchdynamo
import torch

torchdynamo.config.cache_size_limit = 512
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate2 = torchdynamo.optimize("inductor")(model.generate)
torch._dynamo.config.verbose=True
inputs = tokenizer("Generate taxonomy for query: dildo", return_tensors="pt").to('cuda')
# dynamo warm up
with torch.inference_mode():
    outputs = model.generate2(inputs=inputs["input_ids"])