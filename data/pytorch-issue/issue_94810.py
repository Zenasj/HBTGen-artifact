import numpy as np
from onnxruntime import InferenceSession
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("deployment/best_model/")
model = AutoModelForSequenceClassification.from_pretrained("deployment/best_model/")

model.to("cpu")
model.eval()

example_input = tokenizer(
    dataset["test"]["text"][0], max_length=512, truncation=True, return_tensors="pt"
)
_ = model(**example_input)

torch.onnx.export(
    model,
    tuple(example_input.values()),
    f="model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    },
    do_constant_folding=True,
    opset_version=16,
)

session = InferenceSession("deployment/model.onnx", providers=["CPUExecutionProvider"])

y_hat_torch = []
y_hat_onnx = []

for text in dataset["test"]["text"]:
    tok_text = tokenizer(
        text, padding="max_length", max_length=512, truncation=True, return_tensors="np"
    )
    pred = session.run(None, input_feed=dict(tok_text))
    pred = np.argsort(pred[0][0])[::-1][0]
    y_hat_onnx.append(int(pred))

    tok_text = tokenizer(
        text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    pred = model(**tok_text)
    pred = torch.argsort(pred[0][0], descending=True)[0].numpy()
    y_hat_torch.append(int(pred))

print(
    f"Accuracy onnx:{sum([int(i)== int(j) for I, j in zip(y_hat_onnx, dataset['test']['label'])]) / len(y_hat_onnx):.2f}"
)
print(
    f"Accuracy torch:{sum([int(i)== int(j) for I, j in zip(y_hat_torch, dataset['test']['label'])]) / len(y_hat_torch):.2f}"
)

import torch
import onnx
from onnx import numpy_helper

import numpy as np
from numpy.testing import assert_almost_equal

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("deployment/best_model/")
onnx_model = onnx.load("deployment/model.onnx")

graph = onnx_model.graph

initalizers = dict()
for init in graph.initializer:
    initalizers[init.name] = numpy_helper.to_array(init).astype(np.float16)

model_init = dict()
for name, p in model.named_parameters():
    model_init[name] = p.detach().numpy().astype(np.float16)

assert len(initalizers) == len(model_init.keys()) # 53 layers

assert_almost_equal(initalizers['longformer.embeddings.word_embeddings.weight'], 
                    model_init['longformer.embeddings.word_embeddings.weight'], decimal=5)

assert_almost_equal(initalizers['classifier.dense.weight'], 
                    model_init['classifier.dense.weight'], decimal=5)

assert_almost_equal(initalizers['onnx::MatMul_6692'], 
                    model_init['longformer.encoder.layer.0.output.dense.weight'], decimal=4)