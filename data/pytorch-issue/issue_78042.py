3
import torch
import transformers as tr
from datasets import load_dataset

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader

bert = tr.AutoModel.from_pretrained("bert-base-cased")
tokenizer = tr.AutoTokenizer.from_pretrained("bert-base-cased")

datasets = load_dataset("conll2003")
test_data = datasets["test"]


def collate(batch):
    return tokenizer(
        [t["tokens"] for t in batch],
        return_tensors="pt",
        padding=True,
        is_split_into_words=True,
    )


dataloader = DataLoader(
    test_data,
    batch_size=32,
    collate_fn=collate,
    num_workers=0,
)

times_mps = []
bert = bert.to("mps")
# group in batches of 32
for model_inputs in tqdm(dataloader):
    model_inputs = {k: v.to("mps") for k, v in model_inputs.items()}
    t0 = time()
    with torch.no_grad():
        outputs = bert(**model_inputs)
    t1 = time()
    times_mps.append(t1 - t0)

times = []
bert = bert.to("cpu")
# group in batches of 32
for model_inputs in tqdm(dataloader):
    model_inputs = {k: v.to("cpu") for k, v in model_inputs.items()}
    t0 = time()
    with torch.no_grad():
        outputs = bert(**model_inputs)
    t1 = time()
    times.append(t1 - t0)

print("Mean time per batch GPU: {:.2f}s".format(sum(times_mps) / len(times_mps)))
print("Mean time per batch CPU: {:.2f}s".format(sum(times) / len(times)))