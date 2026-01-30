from transformers import AutoModel
from transformers import AutoTokenizer
import torch
model_ckpt = "distilbert-base-uncased"
device = torch.device("mps")
model = AutoModel.from_pretrained(model_ckpt).to(device)
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)

generator = transformers.pipeline(task="text-generation", model=model.to('mps'), tokenizer=tokenizer, device=torch.device("mps"))
generator("This shall brake. ", max_length=200, use_cache=True)