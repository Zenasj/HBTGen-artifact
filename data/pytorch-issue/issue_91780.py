import torch
from torch import _dynamo as dynamo

def backend_fn(gm: torch.fx.GraphModule, example_inputs):
    torchscript = torch.jit.trace(gm, example_inputs)
    example_outputs = torchscript(*example_inputs)
    print("example outputs", example_outputs)

@dynamo.optimize(backend_fn)
def add_one(x):
    return x + 1

x = torch.ones((2, 2))
add_one(x)

import torch
from torch import _dynamo as dynamo
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def backend_fn(gm: torch.fx.GraphModule, example_inputs):
    torchscript = torch.jit.trace(gm, example_inputs, check_trace=False)
    example_outputs = torchscript(*example_inputs)
    print("example outputs", example_outputs)
    return torchscript

model_id = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = DistilBertTokenizer.from_pretrained(model_id)
model = DistilBertForSequenceClassification.from_pretrained(model_id).eval()

examples = [
    "Hello, world!",
    "Goodbye, world!"
]

inputs = tokenizer(examples, return_tensors="pt")

@dynamo.optimize(backend_fn)
def run(inputs):
    return model(**inputs)

with torch.no_grad():
    result = run(inputs)

predicted_class_ids = result.logits.argmax(dim=1)

for id in predicted_class_ids.numpy():
    print(model.config.id2label[id])