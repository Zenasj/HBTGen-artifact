import torch
from torch.export import export_for_training
from transformers import AlbertTokenizer, AlbertModel

print("[info] load model")
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
model = AlbertModel.from_pretrained("albert-base-v1")
model = model.eval()
text = "Hello, how are you?"
example_input = tokenizer(text, return_tensors='pt')  # it is a dict of length 3
with torch.no_grad():
    print("[info] export model")
    exported_model = export_for_training(
        model,
        args=(),
        kwargs=example_input
    ).module() # error here