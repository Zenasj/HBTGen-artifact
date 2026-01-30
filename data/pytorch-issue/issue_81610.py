import torch

tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "distilbert-base-cased",trust_repo=True)
bert = torch.hub.load("huggingface/pytorch-transformers", "model", "distilbert-base-cased",trust_repo=True)
mps_device = torch.device("mps")
bert.to(mps_device)

index_tokens = tokenizer.encode("12345678","12345678", add_special_tokens=True)
tokens_tensor = torch.tensor([index_tokens], device=mps_device)

for i in range(0,2000):
    bert(input_ids=tokens_tensor)
    if i % 100 == 0:
        print(i)