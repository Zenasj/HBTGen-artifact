# MPS Version
from transformers import AutoTokenizer, BertForSequenceClassification
import timeit
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased").to(torch.device("mps"))

tokens = tokenizer.tokenize("Hello world, this is michael!")
tids = tokenizer.convert_tokens_to_ids(tokens)
with torch.no_grad():
    t_tids = torch.tensor([tids]*64, device=torch.device("mps"))
    res = timeit.timeit(lambda: model(input_ids=t_tids), number=100)
print(res)

a_cpu = torch.rand(1000, device='cpu')
b_cpu = torch.rand((1000, 1000), device='cpu')
a_mps = torch.rand(1000, device='mps')
b_mps = torch.rand((1000, 1000), device='mps')

print('cpu', timeit.timeit(lambda: a_cpu @ b_cpu, number=100_000))
print('mps', timeit.timeit(lambda: a_mps @ b_mps, number=100_000))

a_cpu = torch.rand(250, device='cpu')
b_cpu = torch.rand((250, 250), device='cpu')
a_mps = torch.rand(250, device='mps')
b_mps = torch.rand((250, 250), device='mps')

print('cpu', timeit.timeit(lambda: a_cpu @ b_cpu, number=100_000))
print('mps', timeit.timeit(lambda: a_mps @ b_mps, number=100_000))